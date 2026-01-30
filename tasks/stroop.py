"""
Stroop Task (fMRI / Behavioral)
-------------------------------
Refactorisation complète utilisant l'architecture BaseTask.
Gère le Go/No-Go, les couleurs variables et le timing fMRI précis.

Auteur : Clément BARBE / CENIR
Date : Janvier 2026
"""

import random
import sys
import os
import gc
import glob

from psychopy import visual, event, core
from utils.base_task import BaseTask
from utils.utils import should_quit
from tasks.qc.qc_stroop import qc_stroop

class Stroop(BaseTask):
    """
    Tâche Stroop héritant de BaseTask.
    """

    def __init__(self, win, nom, session='01', enregistrer=True, 
                 mode='fmri', n_trials=80, n_choices=4, go_nogo=False,
                 stim_dur=2.0, isi_range=(1500, 2500),
                 parport_actif=True, eyetracker_actif=False, **kwargs):
        
        # 1. INIT PARENT (Gère Logger, Hardware, Chemins, Window)
        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Stroop",
            folder_name="stroop",
            et_prefix="STR",
            eyetracker_actif=eyetracker_actif,
            parport_actif=(mode == 'fmri' and parport_actif), # Sécurité si pas fMRI
            enregistrer=enregistrer
        )

        # 2. PARAMÈTRES SPÉCIFIQUES STROOP
        self.mode = mode
        self.n_trials = n_trials
        self.n_choices = n_choices
        self.go_nogo = go_nogo
        self.stim_dur = stim_dur
        self.isi_range = (isi_range[0]/1000.0, isi_range[1]/1000.0)

        # Validation
        if self.n_choices not in [2, 3, 4]:
            raise ValueError("n_choices doit être 2, 3 ou 4.")
        
        self.logger.log(f"Config Stroop: {self.n_choices} Choix | Go/No-Go: {self.go_nogo}")

        # 3. DÉFINITION DES CODES TTL
        self.codes = {
            'start_exp': 255, 
            'rest_start': 200, 
            'rest_end': 201, 
            'stim_congruent': 10, 
            'stim_incongruent': 11,
            'resp_correct': 100,      # Hit
            'resp_error': 101,        # Wrong Color
            'resp_miss': 102,         # Miss
            'resp_false_alarm': 103,  # Error (No-Go)
            'resp_correct_rej': 104,  # Correct (No-Go)
            'fixation': 5
        }

        # 4. CONFIGURATION DES STIMULI & COULEURS
        # Master Config
        self.MASTER_CONFIG = [
            {'word': 'ROUGE',  'ink': 'red',    'hex': '#FF0000', 'key_fmri': 'b', 'key_behav': 'r'}, # b=blue button box
            {'word': 'VERT',   'ink': 'green',  'hex': '#00FF00', 'key_fmri': 'y', 'key_behav': 'v'}, # y=yellow button box
            {'word': 'BLEU',   'ink': 'blue',   'hex': '#0088FF', 'key_fmri': 'g', 'key_behav': 'b'}, # g=green button box
            {'word': 'JAUNE',  'ink': 'yellow', 'hex': '#FFFF00', 'key_fmri': 'r', 'key_behav': 'j'}  # r=red button box
        ]
        
        # Configuration active (selon nombre de choix)
        self.active_config = self.MASTER_CONFIG[:self.n_choices]
        self.target_inks = set(item['ink'] for item in self.active_config)
        self.colors_hex = {item['ink']: item['hex'] for item in self.MASTER_CONFIG}

        # Setup des touches
        self._setup_keys()

        # Stimuli Spécifiques (Fixation et Instructions sont gérés par BaseTask)
        self.stroop_stim = visual.TextStim(self.win, text='', height=0.15, bold=True)
        
        # Liste pour stockage
        self.global_records = [] 

    def _setup_keys(self):
        """Map les touches selon le mode (fMRI vs Clavier)."""
        self.key_mapping = {}
        for item in self.active_config:
            ink = item['ink']
            # Choix de la touche selon le mode
            key = item['key_fmri'] if self.mode == 'fmri' else item['key_behav']
            self.key_mapping[key] = ink
                
        self.response_keys = list(self.key_mapping.keys())
        self.quit_key = 'escape'

    # =========================================================================
    # LOGIQUE ESSAI (TRIAL)
    # =========================================================================

    def run_trial(self, trial_idx, trial_data):
        """Exécute un essai unique."""
        
        word = trial_data['word']
        ink = trial_data['ink']
        congruent = trial_data['congruent']
        trial_type = trial_data['trial_type'] # 'GO' ou 'NOGO'

        # Préparation
        should_quit(self.win)
        gc.disable() 
        
        self.stroop_stim.text = word
        self.stroop_stim.color = self.colors_hex[ink]
        
        trig_stim = self.codes['stim_congruent'] if congruent else self.codes['stim_incongruent']

        # 1. Fixation
        self.fixation.draw()
        self.win.flip()

        # 2. Stimulus Onset
        self.stroop_stim.draw()
        self.win.callOnFlip(self.ParPort.send_trigger, trig_stim)
        
        if self.eyetracker_actif:
             self.win.callOnFlip(self.EyeTracker.send_message, f"TRIAL_{trial_idx}_STIM")

        self.win.flip() 
        onset_time = self.task_clock.getTime() 
        
        # 3. Réponse
        keys = event.waitKeys(maxWait=self.stim_dur, keyList=self.response_keys + [self.quit_key], timeStamped=self.task_clock)
        
        resp_key = None
        rt = None
        
        # Feedback immédiat
        self.fixation.draw()
        self.win.flip()

        if keys:
            k, t = keys[0] 
            if k == self.quit_key: should_quit(self.win, quit=True)
            resp_key = k
            rt = t - onset_time 
        
        # 4. Analyse Réponse (Scoring)
        acc = 0
        status = "UNKNOWN"
        trig_resp = 0

        if trial_type == 'GO':
            if resp_key:
                user_color = self.key_mapping.get(resp_key)
                if user_color == ink:
                    acc = 1; status = "HIT"; trig_resp = self.codes['resp_correct']
                else:
                    acc = 0; status = "ERROR"; trig_resp = self.codes['resp_error']
            else:
                acc = 0; status = "MISS"; trig_resp = self.codes['resp_miss']

        elif trial_type == 'NOGO':
            if resp_key:
                acc = 0; status = "FALSE_ALARM"; trig_resp = self.codes['resp_false_alarm']
            else:
                acc = 1; status = "CORRECT_REJ"; trig_resp = self.codes['resp_correct_rej']

        # Envoi Trigger Réponse
        self.ParPort.send_trigger(trig_resp)
        if self.eyetracker_actif: self.EyeTracker.send_message(f"RESP_{status}")

        rt_str = f"{rt:.3f}s" if rt else "---"
        cong_str = "CONG" if congruent else "INCONG"
        log_msg = f"T{trial_idx}: {trial_type} | {word}/{ink} ({cong_str}) -> {status} [{rt_str}]"
        
        if status in ["HIT", "CORRECT_REJ"]:
            self.logger.log(log_msg) # En blanc/info standard
        else:
            self.logger.log(log_msg) # En jaune/orange pour les erreurs (plus visible)
        # -----------------------------

        self.global_records.append({
            'participant': self.nom,
            'session': self.session,
            'trial_number': trial_idx,
            'onset_time': onset_time,
            'trial_type': trial_type,
            'word': word,
            'ink': ink,
            'congruent': congruent,
            'response_key': resp_key,
            'rt': rt,
            'accuracy': acc,
            'status': status,
            'trigger_stim': trig_stim,
            'trigger_resp': trig_resp
        })
        
        gc.enable()

        # 6. ISI Jittered
        isi = random.uniform(*self.isi_range)
        self.fixation.draw()
        self.win.callOnFlip(self.ParPort.send_trigger, self.codes['fixation'])
        self.win.flip()
        core.wait(isi)

    # =========================================================================
    # LOGIQUE GÉNÉRALE
    # =========================================================================

    def build_trials(self):
        """Génère la liste pseudo-aléatoire des essais."""
        # Source : Si Go/NoGo, on prend tout le Master Config (car NOGO peut être hors active targets)
        # Sinon on prend juste active_config
        source_config = self.MASTER_CONFIG if self.go_nogo else self.active_config
        
        words_pool = [x['word'] for x in source_config]
        colors_pool = [x['ink'] for x in source_config]
        
        fr_to_eng = {item['word']: item['ink'] for item in self.MASTER_CONFIG}
        base_trials = []
        
        for w in words_pool:
            for ink in colors_pool:
                is_congruent = (fr_to_eng[w] == ink)
                is_go = (ink in self.target_inks) # C'est un GO si la couleur est une cible
                
                base_trials.append({
                    'word': w, 
                    'ink': ink, 
                    'congruent': is_congruent,
                    'trial_type': 'GO' if is_go else 'NOGO'
                })
        
        # Duplication pour atteindre n_trials
        full_trials = base_trials * (self.n_trials // len(base_trials) + 1)
        full_trials = full_trials[:self.n_trials]
        random.shuffle(full_trials)
        
        return full_trials

    def get_instruction_text(self):
        """Génère le texte des consignes."""
        cibles_txt = " / ".join([i['word'] for i in self.active_config])
        
        if self.go_nogo:
            msg = (f"TÂCHE GO / NO-GO\n\n"
                   f"Répondez si l'encre est :\n{cibles_txt}\n\n"
                   f"Si c'est une autre couleur : NE FAITES RIEN.")
        else:
            msg = (f"TÂCHE STROOP\n\n"
                   f"Répondez à la couleur de L'ENCRE.\n"
                   f"Couleurs : {cibles_txt}")
                   
        return msg + "\n\nAppuyez pour commencer."

    def run(self):
        """Méthode principale exécutée par le launcher."""
        try:
            # 1. Génération Trials
            trials = self.build_trials()
            self.logger.log(f"Démarrage Run: {len(trials)} essais.")

            # 2. Instructions (Utilise BaseTask)
            self.show_instructions(self.get_instruction_text())
            
            # 3. Wait Trigger (Utilise BaseTask - Lance EyeTracker & Clock)
            self.wait_for_trigger()
            
            # 4. Baseline Début
            self.show_resting_state(duration_s=10.0, code_start_key='rest_start')
            
            # 5. Boucle Principale
            for i, trial_data in enumerate(trials, 1):
                self.run_trial(i, trial_data)
            
            # 6. Baseline Fin
            self.show_resting_state(duration_s=10.0, code_start_key='rest_end')

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Arrêt manuel par l'utilisateur.")
        except Exception as e:
            self.logger.err(f"Erreur critique Stroop: {e}")
            raise e
        finally:
            # 7. Sauvegarde (Utilise BaseTask)
            self.save_data(self.global_records)

            # Astuce pour trouver le dernier fichier créé si on ne connait pas le nom exact
            list_of_files = glob.glob(os.path.join(self.data_dir, '*.csv')) 
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                
                # --- LANCEMENT DU QC ---
                try:
                    qc_stroop(latest_file)
                except Exception as e:
                    self.logger.warn(f"Echec génération QC: {e}")