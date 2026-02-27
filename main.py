# main.py
import sys
import signal 
from PyQt6.QtWidgets import QApplication
from gui.menu import ExperimentMenu
from utils.logger import get_logger

# Permet de quitter proprement avec Ctrl+C dans le terminal si besoin
signal.signal(signal.SIGINT, signal.SIG_DFL)

def show_menu_and_get_config(app, last_config=None):
    """
    Affiche le menu PyQt et bloque jusqu'à validation.
    """
    menu = ExperimentMenu(last_config)
    menu.show()
    
    app.exec() # Bloque ici tant que la fenêtre est ouverte
    config = menu.get_config()
    
    menu.deleteLater()
    app.processEvents() 
    
    return config

def run_task_logic(config):
    """
    Lance la tâche PsychoPy.
    Nettoyé : La sauvegarde est désormais déléguée à la tâche elle-même via BaseTask.
    """
    logger = get_logger()
    
    # Imports différés pour ne pas charger PsychoPy tant qu'on est dans le menu
    from psychopy import visual, core, logging
    from utils.task_factory import create_task 
    
    # Evite le spam de logs internes de PsychoPy
    logging.console.setLevel(logging.ERROR)
    
    # Création de la fenêtre PsychoPy
    win = visual.Window(
        fullscr=config.get('fullscr', True),
        color='black',
        units='norm',
        screen=config.get('screenid', 0),
        checkTiming=False,
        waitBlanking=True
    )
    
    # On cache la souris
    win.mouseVisible = False

    # Instanciation de la tâche via la Factory
    # Assure-toi que create_task passe bien **config au constructeur !
    task = create_task(config, win)
    
    if not task:
        logger.err(f"Factory Error: Could not create task '{config.get('tache')}'")
        win.close()
        return

    try:
        # Petit temps de calage technique
        win.flip()
        core.wait(0.5) 
        # Lancement de la tâche
        task.run()
        
    except Exception as e:
        logger.err(f"Runtime Error during task execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        win.close()

def main():
    """
    Point d'entrée. Boucle : Menu -> Tâche -> Menu.
    """
    logger = get_logger()
    app = QApplication(sys.argv)
    last_config = None

    while True:
        # 1. Phase Menu (PyQt)
        config = show_menu_and_get_config(app, last_config)

        # Si config est None, l'utilisateur a fermé la croix rouge du menu -> On quitte tout.
        if not config:
            logger.log("Sortie demandée par l'utilisateur.")
            break 
        
        # 2. Phase Exécution (PsychoPy)
        try:
            logger.log(f"Lancement de la tâche : {config.get('tache', 'Unknown')}...")
            
            run_task_logic(config)
            
            # On garde la config en mémoire pour pré-remplir le menu au prochain tour
            last_config = config
            
        except Exception as e:
            logger.err(f"Erreur fatale dans la boucle principale : {e}")
            pass # On continue la boucle pour permettre de relancer

    # Arrêt propre
    logger.log("Application shutdown.")
    app.quit() 
    sys.exit(0)

if __name__ == '__main__':
    main()