"""
BaseTask - Classe mère pour les tâches fMRI/Comportementales
------------------------------------------------------------
Gère l'initialisation commune, le hardware, les chemins et les
fonctions de timing standard (Trigger, Resting State).

Auteur : Clément BARBE / CENIR
Date : Janvier 2026
"""

import os
import csv
from datetime import datetime

from psychopy import visual, core
from psychopy.hardware.keyboard import Keyboard   
from utils.logger import get_logger
from utils.hardware_manager import setup_hardware


class BaseTask:
    """
    Classe mère de toute tâche cognitive.

    API Clavier (3 méthodes) :
        flush_keyboard()  → vide le buffer
        wait_keys()       → bloquant (instructions, trigger, réponses)
        get_keys()        → non-bloquant (polling en boucle)
    """

    QUIT_KEYS = ['escape', 'q']

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
        self.task_clock = core.Clock()
        self.codes = {}

        self._init_paths(folder_name)
        self._init_hardware()
        self._init_keyboard()
        self._init_common_stimuli()

    def should_quit(self, force_quit=False):
        """
        Vérifie si on doit quitter l'expérience (Échap, Q ou A).
        À appeler dans vos boucles (ex: pendant les core.wait ou les win.flip).
        """
        quit_detected = force_quit

        if not quit_detected:
            keys = self.kb.getKeys(keyList=['escape', 'q', 'a'], waitRelease=False, clear=True)
            if keys:
                quit_detected = True

        if quit_detected:
            if self.win:
                try:
                    self.win.close()
                except Exception:
                    pass
            core.quit() 
            return True
        return False

    # =================================================================
    #  CLAVIER — 3 MÉTHODES, C'EST TOUT
    # =================================================================

    def _init_keyboard(self):
        """
        Crée le Keyboard lié à task_clock.
        
        Tous les timestamps (.tDown, .rt) sont référencés 
        à self.task_clock automatiquement.
        """
        self.kb = Keyboard(clock=self.task_clock)
        self.logger.log("Keyboard init (psychtoolbox backend).")

    def flush_keyboard(self):
        """Vide le buffer. À appeler avant chaque phase d'écoute."""
        self.kb.clearEvents()

    def wait_keys(self, key_list=None, max_wait=float('inf')):
        """
        Attente bloquante d'un appui clavier.

        Args:
            key_list: Touches acceptées. None = toutes.
            max_wait: Timeout en secondes. inf = infini.

        Returns:
            list[Key] ou None si timeout.
            Chaque Key a : .name, .tDown, .rt, .duration

        Usage :
            # Instructions → attente infinie, toutes touches
            self.wait_keys()
            
            # Réponse avec timeout
            keys = self.wait_keys(['a','z','e'], max_wait=5.0)
            if keys:
                name, rt = keys[0].name, keys[0].tDown
        """
        self.flush_keyboard()

        keys = self.kb.waitKeys(
            keyList=key_list,
            maxWait=max_wait,
            waitRelease=False,
            clear=True
        )

        if keys and keys[0].name in self.QUIT_KEYS:
            self.logger.warn(f"Quit: {keys[0].name}")
            self.should_quit()

        return keys

    def get_keys(self, key_list=None):
        """
        Lecture non-bloquante du buffer clavier.

        Args:
            key_list: Filtre touches. None = toutes.

        Returns:
            list[Key] (peut être vide).

        Usage dans une boucle de polling :
            while True:
                keys = self.get_keys([self.key_action])
                if keys:
                    action_time = keys[0].tDown
                    break
                core.wait(0.0005)
        """
        keys = self.kb.getKeys(
            keyList=key_list,
            waitRelease=False,
            clear=True
        )

        for k in keys:
            if k.name in self.QUIT_KEYS:
                self.logger.warn(f"Quit: {k.name}")
                self.should_quit(quit=True)

        return keys

    # =================================================================
    #  CHEMINS
    # =================================================================

    def _init_paths(self, folder_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(base_dir) in ('tasks', 'utils'):
            self.root_dir = os.path.dirname(base_dir)
        else:
            self.root_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

        self.data_dir = os.path.join(self.root_dir, 'data', folder_name)
        self.img_dir = os.path.join(self.root_dir, 'image')

        if self.enregistrer and not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
                self.logger.log(f"Dossier créé : {self.data_dir}")
            except OSError as e:
                self.logger.err(f"Erreur dossier : {e}")

    # =================================================================
    #  HARDWARE
    # =================================================================

    def _init_hardware(self):
        self.logger.log(f"Hardware init : {self.task_name}")
        try:
            self.ParPort, self.EyeTracker = setup_hardware(
                self.parport_actif, self.eyetracker_actif, self.win
            )
            if self.eyetracker_actif:
                short_nom = self.nom[:3] if len(self.nom) >= 3 else self.nom
                et_filename = f"{self.et_prefix}_{short_nom}{self.session}"
                if len(et_filename) > 8:
                    et_filename = et_filename[:8]
                    self.logger.warn(f"Nom EDF tronqué : {et_filename}")
                self.EyeTracker.initialize(file_name=et_filename)
        except Exception as e:
            self.logger.err(f"Hardware Error: {e}")
            raise

    # =================================================================
    #  STIMULI COMMUNS
    # =================================================================

    def _init_common_stimuli(self):
        self.fixation = visual.TextStim(
            self.win, text='+', height=0.1, color='white'
        )
        self.instr_stim = visual.TextStim(
            self.win, text='', height=0.06, color='white', wrapWidth=1.5
        )

    # =================================================================
    #  PROTOCOLE STANDARD
    # =================================================================

    def show_instructions(self, text_override=None):
        """Affiche instructions, attend un appui."""
        msg = text_override or (
            f"Bienvenue : {self.task_name}\n\n"
            "Appuyez sur une touche pour continuer."
        )
        self.instr_stim.text = msg
        self.instr_stim.draw()
        self.win.flip()
        core.wait(0.5)
        self.wait_keys()  # bloquant, toutes touches

    def wait_for_trigger(self, trigger_key='t'):
        """Attente trigger IRM → reset clock → start markers."""
        self.instr_stim.text = f"Attente trigger IRM [{trigger_key}]..."
        self.instr_stim.draw()
        self.win.flip()
        self.logger.log("Waiting for trigger...")

        # Attente trigger uniquement (pas de quit check ici)
        self.flush_keyboard()
        self.kb.waitKeys(keyList=[trigger_key], waitRelease=False, clear=True)

        # t=0
        self.task_clock.reset()

        start_code = self.codes.get('start_exp', 255)
        self.ParPort.send_trigger(start_code)

        if self.eyetracker_actif:
            self.EyeTracker.start_recording()
            self.EyeTracker.send_message(f"START_{self.task_name.upper()}")

        self.logger.ok(f"Trigger reçu. Code: {start_code}")

    def show_resting_state(self, duration_s=10.0,
                           code_start_key='rest_start',
                           code_end_key='rest_end'):
        """Fixation pour baseline."""
        self.logger.log(f"Rest : {duration_s}s")

        self.ParPort.send_trigger(self.codes.get(code_start_key, 0))
        if self.eyetracker_actif:
            self.EyeTracker.send_message("REST_START")

        self.fixation.draw()
        self.win.flip()
        core.wait(duration_s)

        if code_end_key:
            self.ParPort.send_trigger(self.codes.get(code_end_key, 0))
            if self.eyetracker_actif:
                self.EyeTracker.send_message("REST_END")

    # =================================================================
    #  SAUVEGARDE
    # =================================================================

    def save_data(self, data_list=None, filename_suffix=""):
        if data_list is None:
            data_list = getattr(self, 'global_records', [])

        if not self.enregistrer or not data_list:
            self.logger.warn("Rien à sauvegarder.")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self.task_name.replace(' ', '')
        fname = f"{self.nom}_{safe_name}{filename_suffix}_{timestamp}.csv"
        path = os.path.join(self.data_dir, fname)

        try:
            all_keys = sorted(set().union(*(d.keys() for d in data_list)))
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(data_list)
            self.logger.ok(f"Saved : {path}")
        except Exception as e:
            self.logger.err(f"SAVE ERROR : {e}")
            with open(path + '.bak', 'w') as f:
                f.write(str(data_list))

        return path

    # =================================================================
    #  INTERFACE
    # =================================================================

    def run(self):
        raise NotImplementedError("Implémenter run() dans la sous-classe.")