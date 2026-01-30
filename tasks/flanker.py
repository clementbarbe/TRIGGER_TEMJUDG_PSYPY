import os
import sys
import random
import gc
import glob

from psychopy import visual, event, core
from utils.base_task import BaseTask
from utils.utils import should_quit
from tasks.qc.qc_flanker import qc_flanker


class Flanker(BaseTask):
    """
    Flanker Task - Rapid Event-Related fMRI
    - ASCII only (<, >)
    - Full 3 flankers (3 de chaque côté => 7 symboles)
    - Pas de gras
    - Stimulus un peu plus long (stim_dur)
    - Logs courts par trial (congruent/incongruent + bon/mauvais + RT)
    """

    def __init__(self, win, nom, session='01', enregistrer=True,
                 mode='fmri', n_trials=80,
                 stim_dur=0.75,              # <- augmenté (avant 0.5)
                 response_window=1.5,        # <- inchangé ici
                 isi_min=1.0, isi_max=5.0, isi_mean=2.5,
                 parport_actif=True, eyetracker_actif=False,
                 font_name='DejaVu Sans Mono',
                 show_trial_logs=True,       # <- active les petits logs
                 **kwargs):

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Flanker",
            folder_name="flanker",
            eyetracker_actif=eyetracker_actif,
            parport_actif=(mode == 'fmri' and parport_actif),
            enregistrer=enregistrer,
            et_prefix="FLK"
        )

        self.n_trials = n_trials
        self.stim_dur = stim_dur
        self.resp_window = response_window
        self.isi_params = (isi_min, isi_max, isi_mean)
        self.font_name = font_name
        self.show_trial_logs = show_trial_logs

        self.codes = {
            'start_exp': 255,
            'stim_congruent': 11,
            'stim_incongruent': 12,
            'resp_left': 1,
            'resp_right': 2,
            'fixation': 5
        }

        self.stim_text = visual.TextStim(
            self.win, text='', color='white',
            height=0.15, font=self.font_name, bold=False
        )

        self._stim_cache = {}

        if mode == 'fmri':
            self.keys = {'left': 'b', 'right': 'y'}
        else:
            self.keys = {'left': 'left', 'right': 'right'}

        self.quit_key = 'escape'
        self.trials_design = []
        self.global_records = []

    # ---------- Stimulus helpers (ASCII only) ----------
    def _build_flanker_string(self, target, condition):
        symbols = {'left': '<', 'right': '>'}

        if condition == 'congruent':
            flank = target
        else:
            flank = 'right' if target == 'left' else 'left'

        flank_sym = symbols[flank]
        targ_sym = symbols[target]

        return f"{flank_sym * 3}{targ_sym}{flank_sym * 3}"  # 7 caractères

    def _get_stim(self, stim_str):
        stim = self._stim_cache.get(stim_str)
        if stim is None:
            stim = visual.TextStim(
                self.win, text=stim_str, color='white',
                height=0.15, font=self.font_name, bold=False
            )
            self._stim_cache[stim_str] = stim
        return stim

    # ---------- Design ----------
    def generate_design(self):
        isi_min, isi_max, isi_mean = self.isi_params
        lam = 1.0 / isi_mean

        isis = []
        while len(isis) < self.n_trials:
            sample = random.expovariate(lam)
            if isi_min <= sample <= isi_max:
                isis.append(round(sample, 3))

        conds = (['congruent'] * (self.n_trials // 2) +
                 ['incongruent'] * (self.n_trials // 2))
        random.shuffle(conds)

        for i in range(self.n_trials):
            target = random.choice(['left', 'right'])
            stim_str = self._build_flanker_string(target, conds[i])

            self.trials_design.append({
                'stimulus': stim_str,
                'target': target,
                'condition': conds[i],
                'isi': isis[i],
                'n_flank': 3
            })

        # Warm cache (évite coût inattendu en run)
        for s in {t['stimulus'] for t in self.trials_design}:
            _ = self._get_stim(s)

    def _wait_until(self, t_goal, relax=0.001):
        while self.task_clock.getTime() < t_goal:
            dt = t_goal - self.task_clock.getTime()
            core.wait(min(relax, dt))

    def run_trial(self, trial_idx, trial_data, onset_goal):
        should_quit(self.win)
        gc.disable()
        event.clearEvents(eventType='keyboard')

        trig_stim = self.codes[f"stim_{trial_data['condition']}"]

        self._wait_until(onset_goal - 0.012)

        stim_obj = self._get_stim(trial_data['stimulus'])

        # STIM ONSET
        stim_obj.draw()
        self.win.callOnFlip(self.ParPort.send_trigger, trig_stim)
        self.win.flip()
        onset_time = self.task_clock.getTime()

        t_stim_off = onset_time + self.stim_dur
        t_resp_end = onset_time + self.resp_window
        next_onset_anchor = onset_goal + self.resp_window + trial_data['isi']

        resp_key = None
        rt = None

        while self.task_clock.getTime() < t_resp_end:
            now = self.task_clock.getTime()

            if now < t_stim_off:
                stim_obj.draw()
            else:
                self.fixation.draw()

            self.win.flip()

            if resp_key is None:
                keys = event.getKeys(
                    keyList=[self.keys['left'], self.keys['right'], self.quit_key],
                    timeStamped=self.task_clock
                )
                if keys:
                    k, t = keys[0]
                    if k == self.quit_key:
                        should_quit(self.win, quit=True)
                    resp_key = k
                    rt = t - onset_time
                    self.ParPort.send_trigger(
                        self.codes[f"resp_{'left' if k == self.keys['left'] else 'right'}"]
                    )

        correct_key = self.keys[trial_data['target']]
        acc = 1 if resp_key == correct_key else 0

        # ---- Petit log “style demandé” adapté au Flanker ----
        if self.show_trial_logs:
            trial_type = trial_data['condition']  # congruent / incongruent
            # "word/ink" n'existe pas ici; on mappe vers stimulus/target
            word = trial_data['stimulus']
            ink = trial_data['target']
            cong_str = "congruent" if trial_data['condition'] == "congruent" else "incongruent"
            status = "bon" if acc == 1 else "mauvais"
            rt_str = "NA" if rt is None else f"{rt:.3f}s"

            log_msg = f"T{trial_idx+1}: {trial_type} | {word}/{ink} ({cong_str}) -> {status} [{rt_str}]"
            print(log_msg)  # volontairement simple et court (console)

        self.global_records.append({
            'trial_idx': trial_idx + 1,
            'condition': trial_data['condition'],
            'target': trial_data['target'],
            'n_flank': 3,
            'stimulus': trial_data['stimulus'],
            'onset_goal': onset_goal,
            'onset_time': onset_time,
            'rt': rt,
            'acc': acc,
            'isi_jitter': trial_data['isi']
        })

        gc.enable()
        return next_onset_anchor

    def run(self):
        try:
            self.generate_design()

            # Instructions remises au début
            instr = (
                "Tâche des flèches (Flanker)\n\n"
                "Objectif : répondre à la DIRECTION de la flèche CENTRALE.\n\n"
                f"Réponse GAUCHE : {self.keys['left']}\n"
                f"Réponse DROITE : {self.keys['right']}\n\n"
                "Répondez le plus vite et le plus correctement possible.\n"
                "Gardez les yeux au centre (croix de fixation).\n\n"
                "Appuyez sur une touche pour commencer."
            )
            self.show_instructions(instr)

            self.wait_for_trigger()
            self.show_resting_state(5.0)

            next_onset = self.task_clock.getTime() + 0.5

            for i, trial_data in enumerate(self.trials_design):
                next_onset = self.run_trial(i, trial_data, next_onset)

            self.show_resting_state(5.0)

        finally:
            self.save_data(self.global_records)
            list_of_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                try:
                    qc_flanker(latest_file)
                except Exception:
                    pass