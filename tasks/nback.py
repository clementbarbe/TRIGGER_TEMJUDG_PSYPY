import os
import sys
import random
import gc
import glob

from psychopy import visual, event, core
from utils.base_task import BaseTask
from utils.utils import should_quit
from tasks.qc.qc_nback import qc_nback

class NBack(BaseTask):
    """
    N-Back Task (fMRI / Behavioral) - version corrigée

    Changements majeurs:
    - Instructions: minimalistes, petite taille, durée fixe 3s, sans appui touche
    - Mode increm=True: blocs 1-back puis 2-back puis ... jusqu'à N_max
      et *n_trials par bloc* (donc total = n_trials * nb_blocs)
    - Robustesse: triggers ParPort conditionnels (si parport_actif False)
    - Fixation: utilise BaseTask.fixation (déjà initialisée), pas de self.fixation manquant
    - Timing: ancrage par onset_goal (drift control) conservé
    """

    def __init__(self, win, nom, session='01', enregistrer=True,
                 mode='fmri', N=2, n_trials=30, target_ratio=0.3,
                 stim_dur=0.5, isi=1.5,
                 increm=False, advance_frame=True,
                 parport_actif=True, eyetracker_actif=False, **kwargs):

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name=f"NBack_{int(N)}{'_Inc' if increm else ''}",
            folder_name="nback",
            eyetracker_actif=eyetracker_actif,
            parport_actif=(mode == 'fmri' and parport_actif),
            enregistrer=enregistrer,
            et_prefix="NBK"
        )

        # ----------------------------
        # Config
        # ----------------------------
        self.mode = mode
        self.N_max = int(N)
        self.n_trials = int(n_trials)           # IMPORTANT: n_trials = par bloc
        self.target_ratio = float(target_ratio)
        self.stim_dur = float(stim_dur)
        self.isi = float(isi)
        self.increm = bool(increm)
        self.advance_frame = bool(advance_frame)

        # Instructions: 3s fixes, minimal
        self.instr_dur = 3.0

        # ----------------------------
        # Codes triggers
        # ----------------------------
        self.codes = {
            'start_exp': 255,
            'rest_start': 200,
            'rest_end': 201,
            'stim_target': 10,
            'stim_nontarget': 20,
            'resp': 128,
            'fixation': 5
        }

        # ----------------------------
        # Stimuli
        # BaseTask fournit: self.fixation (height=0.1) et self.instr_stim (height=0.06)
        # Ici on ajoute un TextStim dédié à la lettre, bien lisible.
        # ----------------------------
        self.letter_stim = visual.TextStim(
            self.win, text='', color='white',
            height=0.18, font='Arial'
        )

        # Réponses
        if self.mode == 'fmri':
            self.response_keys = ['b', 'y', 'g', 'r']
        else:
            self.response_keys = ['space', 'return', 'a', 'z']
        self.quit_key = 'escape'

        self.global_records = []

        # ----------------------------
        # TIMING GLOBAL (figé)
        # ----------------------------
        # Mesuré UNE SEULE FOIS pour éviter toute instabilité pendant la tâche
        try:
            fr = self.win.getActualFrameRate()
            if fr is None or fr <= 0:
                raise RuntimeError("Frame rate non mesurable")
            self.frame_rate = float(round(fr))
        except Exception:
            # fallback sûr IRMf
            self.frame_rate = 60.0

        self.frame_duration = 1.0 / self.frame_rate
        self.frame_tolerance = self.frame_duration / 2.0

        self.logger.log(
            f"Timing figé: frame_rate={self.frame_rate:.2f} Hz | "
            f"frame_duration={self.frame_duration*1000:.2f} ms"
        )

    # ======================================================================
    # TIMING UTIL
    # ======================================================================

    def _wait_until(self, t_goal, relax=0.002):
        """Attente active légère jusqu'à t_goal (horloge task_clock)."""
        while True:
            now = self.task_clock.getTime()
            dt = t_goal - now
            if dt <= 0:
                return
            core.wait(min(relax, dt))

    def _send_trigger_safe(self, code):
        """Envoie un trigger si ParPort disponible/actif."""
        if getattr(self, "ParPort", None) is None:
            return
        try:
            self.ParPort.send_trigger(int(code))
        except Exception:
            # On évite de planter une séance si le port parallèle bugge
            pass

    def _record_onset(self):
        """Fonction appelée via win.callOnFlip pour marquer précisément l'onset."""
        # on lit l'horloge de la tâche juste après le flip
        self._last_onset_time = self.task_clock.getTime()

    



    # ======================================================================
    # GENERATION SEQUENCE
    # ======================================================================

    def generate_block_sequence(self, n_level, n_trials_block):
        """
        Génère une séquence (letter, is_target) en garantissant que
        la proportion de targets par bloc est comprise entre 30% et 40%.
        Contraintes: un target à l'essai i = lettre identique à i-n_level.
        """
        import math
        letters = list("BCDFGHJKLMNPQRSTVXZ")
        sequence = []

        # bornes 30% - 40%
        min_t = math.ceil(0.30 * n_trials_block)
        max_t = math.floor(0.40 * n_trials_block)

        # cible souhaitée selon target_ratio mais clampée aux bornes
        desired = int(round(self.target_ratio * n_trials_block))
        desired = max(min_t, min(max_t, desired))

        # positions potentielles de target (i >= n_level)
        possible_positions = list(range(n_level, n_trials_block))
        if len(possible_positions) < desired:
            # si pas assez d'emplacements possibles, on réduit
            desired = len(possible_positions)

        # choisir aléatoirement les indices de target dans le bloc
        target_positions = set(random.sample(possible_positions, desired)) if desired > 0 else set()

        history = []
        for i in range(n_trials_block):
            can_be_target = (i >= n_level) and (i in target_positions)
            if can_be_target:
                is_target = True
                # la lettre cible doit être identique à i-n_level (déjà présente dans history)
                letter = history[i - n_level]
            else:
                is_target = False
                if i >= n_level:
                    forbidden = history[i - n_level]
                    pool = [x for x in letters if x != forbidden]
                else:
                    pool = letters
                letter = random.choice(pool)

            history.append(letter)
            sequence.append((letter, is_target))

        # sécurité: si par hasard le ratio ne tombe pas dans la fourchette (improbable), on loggue
        actual_targets = sum(1 for _, t in sequence if t)
        actual_ratio = actual_targets / float(n_trials_block)
        if not (0.30 <= actual_ratio <= 0.40):
            self.logger.warn(f"Target ratio hors bornes: {actual_targets}/{n_trials_block} = {actual_ratio:.2f}")

        return sequence


    # ======================================================================
    # TRIAL EXECUTION
    # ======================================================================

    def run_trial(self, trial_idx_global, letter, is_target, current_N, onset_goal):
        should_quit(self.win)
        gc.disable()
        event.clearEvents(eventType='keyboard')

        trig_stim = self.codes['stim_target'] if is_target else self.codes['stim_nontarget']

        wait_target = onset_goal - self.frame_tolerance
        if self.advance_frame:
            wait_target -= self.frame_duration


        # attente active jusqu'au moment calculé
        self._wait_until(wait_target)

        # --- Stim onset ---
        self.letter_stim.text = letter
        self.letter_stim.draw()

        # Préparer l'envoi de trigger ET la capture d'onset au moment du flip
        # callOnFlip exécute juste après le buffer swap (très précis).
        if getattr(self, "ParPort", None) is not None:
            self.win.callOnFlip(self._send_trigger_safe, trig_stim)

        # Marquer l'onset sur l'horloge de la tâche au flip
        self.win.callOnFlip(self._record_onset)

        # flip: le _record_onset sera appelé au moment du swap
        self.win.flip()

        # Récupère le timestamp que _record_onset a stocké ; sinon fallback immédiat
        onset_time = getattr(self, '_last_onset_time', None)
        if onset_time is None:
            onset_time = self.task_clock.getTime()

        t_stim_end = onset_time + self.stim_dur
        t_trial_end_absolute = onset_goal + self.stim_dur + self.isi

        resp_key = None
        rt = None

        # Réponse pendant le stimulus
        while self.task_clock.getTime() < t_stim_end and resp_key is None:
            keys = event.getKeys(
                keyList=self.response_keys + [self.quit_key],
                timeStamped=self.task_clock
            )
            if keys:
                k, t = keys[0]
                if k == self.quit_key:
                    should_quit(self.win, quit=True)
                resp_key = k
                rt = t - onset_time

        # Fixation
        self.fixation.draw()
        if getattr(self, "ParPort", None) is not None:
            self.win.callOnFlip(self._send_trigger_safe, self.codes['fixation'])
        self.win.flip()

        # Réponse pendant ISI
        while self.task_clock.getTime() < t_trial_end_absolute and resp_key is None:
            keys = event.getKeys(
                keyList=self.response_keys + [self.quit_key],
                timeStamped=self.task_clock
            )
            if keys:
                k, t = keys[0]
                if k == self.quit_key:
                    should_quit(self.win, quit=True)
                resp_key = k
                rt = t - onset_time

        # Si réponse, on attend la fin stricte du trial pour coller à l'onset_goal
        if resp_key is not None:
            self._wait_until(t_trial_end_absolute)

        # Scoring
        if resp_key is None:
            acc = 0 if is_target else 1
            status = "MISS" if is_target else "CR"
            trig_resp = 0
        else:
            acc = 1 if is_target else 0
            status = "HIT" if is_target else "FA"
            trig_resp = self.codes['resp']
            self._send_trigger_safe(trig_resp)

        rt_str = f"{rt:.3f}s" if rt is not None else "---"
        self.logger.log(f"T{trial_idx_global:03d} (N={current_N}) | {letter} | {status} | RT:{rt_str}")

        self.global_records.append({
            'participant': self.nom,
            'session': self.session,
            'task_name': self.task_name,
            'mode': self.mode,

            'trial_number': trial_idx_global,
            'block_N_level': current_N,
            'is_increm': self.increm,

            'letter': letter,
            'is_target': bool(is_target),

            'onset_goal': float(onset_goal),
            'onset_time': float(onset_time),

            'stim_dur': float(self.stim_dur),
            'isi': float(self.isi),

            'rt': None if rt is None else float(rt),
            'resp_key': resp_key,
            'accuracy': int(acc),
            'status': status,

            'trigger_stim': int(trig_stim),
            'trigger_resp': int(trig_resp) if trig_resp else 0
        })

        gc.enable()


    # ======================================================================
    # RUN PRINCIPAL
    # ======================================================================

    def get_instruction_for_level(self, n):
        """Instruction minimaliste: juste l'indication du niveau."""
        return f"{n}-BACK"

    def run(self):
        try:
            # Définition des blocs
            if self.increm:
                # 1-back -> 2-back -> ... -> N_max
                levels = list(range(1, int(self.N_max) + 1))
            else:
                levels = [int(self.N_max)]

            # IMPORTANT: n_trials = par bloc, pas divisé
            trials_per_block = int(self.n_trials)
            total_trials = trials_per_block * len(levels)

            self.logger.log(
                f"Config: Increm={self.increm}, Levels={levels}, Trials/Block={trials_per_block}, TotalTrials={total_trials}, "
                f"stim_dur={self.stim_dur}, isi={self.isi}, target_ratio={self.target_ratio}"
            )

            global_trial_counter = 1

            # Boucle sur blocs
            for i_block, n_level in enumerate(levels):
                # A) Séquence bloc
                block_sequence = self.generate_block_sequence(n_level, trials_per_block)

                # B) Instructions auto (3s)
                self.instr_stim.text = self.get_instruction_for_level(n_level)
                self.instr_stim.draw()
                self.win.flip()
                core.wait(self.instr_dur)

                # C) Sync IRMf: seulement au début du 1er bloc
                if i_block == 0 :
                    self.wait_for_trigger()  # reset clock + start_exp trigger + ET start

                # D) Baseline avant bloc (si tu veux garder)
                self.show_resting_state(duration_s=5.0, code_start_key='rest_start', code_end_key='rest_end')

                # E) Initialisation timing bloc
                start_anchor = self.task_clock.getTime() + 0.5
                trial_len = self.stim_dur + self.isi

                self.fixation.draw()
                self.win.flip()

                # F) Trials
                for i, (letter, is_target) in enumerate(block_sequence, 1):
                    onset_goal = start_anchor + (i - 1) * trial_len
                    self.run_trial(
                        trial_idx_global=global_trial_counter,
                        letter=letter,
                        is_target=is_target,
                        current_N=n_level,
                        onset_goal=onset_goal
                    )
                    global_trial_counter += 1

            # Fin: baseline post
            self.show_resting_state(duration_s=10.0, code_start_key='rest_start', code_end_key='rest_end')

        except Exception as e:
            self.logger.err(f"Erreur NBack: {e}")
            raise

        finally:
            # Sauvegarde
            self.save_data(self.global_records)

            # QC auto sur dernier csv
            list_of_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                try:
                    print(f"Lancement du QC sur : {latest_file}")
                    qc_nback(latest_file)
                except Exception as e:
                    print(f"Erreur lors du QC automatique : {e}")