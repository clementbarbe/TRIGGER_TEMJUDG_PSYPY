"""
BaseTask - Classe mère pour les tâches fMRI/Comportementales
------------------------------------------------------------
Gère l'initialisation commune, le hardware, les chemins et les
fonctions de timing standard (Trigger, Resting State).

Auteur : Clément BARBE / CENIR
Date : Janvier 2026
"""

import os
from datetime import datetime
from psychopy import visual, core
from psychopy.hardware import keyboard
from utils.logger import get_logger
from utils.hardware_manager import setup_hardware


class BaseTask:
    def __init__(self, win, nom, session, task_name, folder_name,
                 eyetracker_actif=False, parport_actif=False,
                 enregistrer=True, et_prefix='TSK'):
        self.win = win
        self.nom = str(nom)
        self.session = str(session)
        self.task_name = task_name
        self.et_prefix = et_prefix

        self.eyetracker_actif = eyetracker_actif
        self.parport_actif = parport_actif
        self.enregistrer = enregistrer

        self.logger = get_logger()

        self._init_paths(folder_name)
        self._init_hardware()
        self._init_common_stimuli()

        self.task_clock = core.Clock()
        self.kb = keyboard.Keyboard(clock=self.task_clock)

        self.codes = {}

    def _init_paths(self, folder_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if os.path.basename(base_dir) == 'tasks':
            self.root_dir = os.path.dirname(base_dir)
        else:
            self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.data_dir = os.path.join(self.root_dir, 'data', folder_name)
        self.img_dir = os.path.join(self.root_dir, 'image')

        if self.enregistrer and not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
                self.logger.log(f"Dossier créé : {self.data_dir}")
            except OSError as e:
                self.logger.err(f"Erreur création dossier data : {e}")

    def _init_hardware(self):
        self.logger.log(f"Setup Hardware pour {self.task_name}...")

        try:
            self.ParPort, self.EyeTracker = setup_hardware(
                self.parport_actif,
                self.eyetracker_actif,
                self.win
            )

            if self.eyetracker_actif:
                short_nom = self.nom[:3] if len(self.nom) >= 3 else self.nom
                et_filename = f"{self.et_prefix}_{short_nom}{self.session}"

                if len(et_filename) > 8:
                    et_filename = et_filename[:8]
                    self.logger.warn(f"Nom fichier ET tronqué à : {et_filename}")

                self.EyeTracker.initialize(file_name=et_filename)

        except Exception as e:
            self.logger.err(f"Hardware Init Error: {e}")
            raise

    def _init_common_stimuli(self):
        self.fixation = visual.TextStim(self.win, text='+', height=0.1, color='white')
        self.instr_stim = visual.TextStim(self.win, text='', height=0.06, color='white', wrapWidth=1.5)

    def kb_clear(self):
        try:
            self.kb.clearEvents()
        except Exception:
            pass

    def kb_get_key(self, keyList=None, clear=True):
        keys = self.kb.getKeys(keyList=keyList, waitRelease=False, clear=clear)
        if not keys:
            return None
        k = keys[0]
        return k.name, k.rt, k.tDown

    def kb_wait_key(self, keyList=None, maxWait=None, clear=True):
        if clear:
            self.kb_clear()
        keys = self.kb.waitKeys(maxWait=maxWait, keyList=keyList, waitRelease=False, clear=False)
        if not keys:
            return None
        k = keys[0]
        return k.name, k.rt, k.tDown

    def show_instructions(self, text_override=None, min_wait_s=0.5):
        msg = text_override if text_override else (
            f"Bienvenue dans la tâche : {self.task_name}\n\n"
            "Appuyez sur une touche pour voir les consignes spécifiques."
        )

        self.instr_stim.text = msg
        self.instr_stim.draw()
        self.win.flip()

        core.wait(min_wait_s)
        self.kb_wait_key(keyList=None, maxWait=None, clear=True)

    def wait_for_trigger(self, trigger_key='t'):
        self.instr_stim.text = "En attente du trigger IRM..."
        self.instr_stim.draw()
        self.win.flip()

        self.logger.log("Waiting for trigger...")

        k = self.kb_wait_key(keyList=[trigger_key], maxWait=None, clear=True)
        while k is None:
            k = self.kb_wait_key(keyList=[trigger_key], maxWait=None, clear=True)

        self.task_clock.reset()
        self.kb = keyboard.Keyboard(clock=self.task_clock)

        start_code = self.codes.get('start_exp', 255)
        if self.parport_actif:
            self.ParPort.send_trigger(start_code)

        if self.eyetracker_actif:
            self.EyeTracker.start_recording()
            self.EyeTracker.send_message(f"START_{self.task_name.upper()}")

        self.logger.log(f"Trigger reçu. Start Code: {start_code}")

    def show_resting_state(self, duration_s=10.0, code_start_key='rest_start', code_end_key='rest_end'):
        self.logger.log(f"Resting state: {duration_s}s")

        c_start = self.codes.get(code_start_key, 0)
        if self.parport_actif:
            self.ParPort.send_trigger(c_start)
        if self.eyetracker_actif:
            self.EyeTracker.send_message("REST_START")

        self.fixation.draw()
        self.win.flip()

        core.wait(duration_s)

        if code_end_key:
            c_end = self.codes.get(code_end_key, 0)
            if self.parport_actif:
                self.ParPort.send_trigger(c_end)
            if self.eyetracker_actif:
                self.EyeTracker.send_message("REST_END")

    def save_data(self, data_list=None, filename_suffix=""):
        if data_list is None:
            data_list = getattr(self, 'global_records', [])

        if not self.enregistrer or not data_list:
            self.logger.warn("Aucune donnée à sauvegarder (ou enregistrement désactivé).")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{self.nom}_{self.task_name.replace(' ', '')}{filename_suffix}_{timestamp}.csv"
        path = os.path.join(self.data_dir, fname)

        try:
            import csv
            keys = set().union(*(d.keys() for d in data_list))

            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(list(keys)))
                writer.writeheader()
                writer.writerows(data_list)
            self.logger.log(f"Data saved: {path}")

        except Exception as e:
            self.logger.err(f"CRITICAL SAVE ERROR: {e}")
            with open(path + '.bak', 'w', encoding='utf-8') as f:
                f.write(str(data_list))