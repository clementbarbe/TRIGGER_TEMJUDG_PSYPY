import random
import gc, os
import glob
from psychopy import visual, event, core
from psychopy.hardware import keyboard
from utils.base_task import BaseTask
from utils.utils import should_quit
from tasks.qc.qc_temporal import qc_temporaljudgement


class TemporalJudgement(BaseTask):
    def __init__(self, win, nom, session='01', mode='fmri', run_type='base',
                 n_trials_base=72, n_trials_block=24, n_trials_training=12,
                 delays_ms=(200, 300, 400, 500, 600, 700),
                 response_options=(100, 200, 300, 400, 500, 600, 700, 800),
                 stim_isi_range=(1500, 2500),
                 enregistrer=True, eyetracker_actif=False, parport_actif=True,
                 passive_action_jitter_s=(0.6, 1.2),
                 response_deadline_s=5.0,
                 gc_collect_every=10,
                 **kwargs):

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Temporal Judgement",
            folder_name="temporal_judgement",
            eyetracker_actif=eyetracker_actif,
            parport_actif=parport_actif,
            enregistrer=enregistrer,
            et_prefix='TJ'
        )

        self.mode = mode.lower()
        self.run_type = run_type.lower()

        self.n_trials_base = n_trials_base
        self.n_trials_block = n_trials_block
        self.n_trials_training = n_trials_training

        self.delays_ms = list(delays_ms)
        self.response_values_ms = list(response_options)
        self.stim_isi_range = (stim_isi_range[0] / 1000.0, stim_isi_range[1] / 1000.0)

        self.passive_action_jitter_s = passive_action_jitter_s
        self.response_deadline_s = float(response_deadline_s)
        self.gc_collect_every = int(gc_collect_every)

        self.global_records = []
        self.current_trial_idx = 0
        self.current_phase = 'setup'

        self._detect_display_scaling()
        self._measure_frame_rate()

        self._define_ttl_codes()

        self.kb = keyboard.Keyboard(clock=self.task_clock)

        self._setup_key_mapping()
        self._setup_task_stimuli()
        self._setup_feedback_stimuli()

        self._gc_counter = 0

        self.logger.ok(f"TemporalJudgement init | Mode: {self.run_type} | Frame Rate: {self.frame_rate:.2f} Hz")

    def _detect_display_scaling(self):
        if self.win.size[1] > 1200:
            self.pixel_scale = 2.0
            self.logger.log(f"Écran Haute Résolution détecté ({self.win.size}). Scale: x2.0")
        else:
            self.pixel_scale = 1.0
            self.logger.log(f"Écran Standard ({self.win.size}). Scale: x1.0")

        self.x_spacing_scale = 1.14

    def _measure_frame_rate(self):
        self.logger.log("Mesure du frame rate en cours...")

        self.frame_rate = self.win.getActualFrameRate(nIdentical=10, nMaxFrames=100, threshold=1)

        if self.frame_rate is None:
            self.frame_rate = 60.0
            self.logger.warn("Frame rate non détecté, valeur par défaut : 60.0 Hz")
        else:
            self.logger.ok(f"Frame rate mesuré : {self.frame_rate:.2f} Hz")

        self.frame_duration_s = 1.0 / self.frame_rate
        self.frame_tolerance_s = 0.75 * self.frame_duration_s
        self.logger.log(f"Frame tolerance : {self.frame_tolerance_s*1000:.2f} ms")

    def _define_ttl_codes(self):
        self.codes = {
            'start_exp': 255,
            'rest_start': 200,
            'rest_end': 201,
            'trial_active': 110,
            'trial_passive': 111,
            'action_bulb': 120,
            'bulb_on': 130,
            'response_prompt': 135,
            'response_given': 140,
            'timeout': 199,
            'crisis_prompt': 150,
            'crisis_start': 151,
            'crisis_end': 152,
            'crisis_valid_prompt': 153,
            'crisis_res_success': 154,
            'crisis_res_fail': 155,
            'crisis_retry_yes': 156,
            'crisis_retry_no': 157
        }

    def _setup_key_mapping(self):
        if self.mode == 'fmri':
            self.key_action = 'b'
            self.keys_responses = ['d', 'n', 'z', 'e', 'b', 'y', 'g', 'r']
            self.key_trigger = 't'
        else:
            self.key_action = 'y'
            self.keys_responses = ['a', 'z', 'e', 'r', 'y', 'u', 'i', 'o']
            self.key_trigger = 't'

        self.keys_quit = ['escape', 'q']

        self.response_key_to_ms = {
            key: ms for key, ms in zip(self.keys_responses, self.response_values_ms)
        }

        self.logger.log(f"Mapping touches ({self.mode}): Action={self.key_action}, Responses={self.keys_responses[:3]}...")

    def _setup_task_stimuli(self):
        bulb_size = (0.45 * 0.9, 0.9 * 0.9)
        bulb_pos = (0.0, 0.0)

        img_off = os.path.join(self.img_dir, 'bulbof.png')
        img_on = os.path.join(self.img_dir, 'bulbon.png')

        if os.path.exists(img_off) and os.path.exists(img_on):
            self.bulb_off_img = visual.ImageStim(
                self.win, image=img_off, size=bulb_size, pos=bulb_pos
            )
            self.bulb_on_img = visual.ImageStim(
                self.win, image=img_on, size=bulb_size, pos=bulb_pos
            )
        else:
            self.logger.warn("Images ampoules absentes, utilisation de cercles.")
            self.bulb_off_img = visual.Circle(self.win, radius=0.2, fillColor='grey')
            self.bulb_on_img = visual.Circle(self.win, radius=0.2, fillColor='yellow')

        self.colored_bar = visual.Rect(
            win=self.win,
            width=0.15,
            height=0.04,
            pos=(0.0, -0.5)
        )

        self.response_title = visual.TextStim(
            self.win,
            text="Combien de ms avez-vous perçu ?",
            color='white',
            height=0.05,
            pos=(0, 0.3)
        )

        self.response_options_text = visual.TextStim(
            self.win,
            text="1: 100 | 2: 200 | 3: 300 | 4: 400 | 5: 500 | 6: 600 | 7: 700 | 8: 800",
            color='white',
            height=0.05,
            pos=(0, 0.05)
        )

        self.response_instr = visual.TextStim(
            self.win,
            text="Répondez avec les 8 boutons",
            color='white',
            height=0.045,
            pos=(0, -0.2)
        )

        base_positions = [-0.35, -0.255, -0.15, -0.05, 0.055, 0.16, 0.26, 0.36]
        self.underline_x_positions = [x * self.x_spacing_scale for x in base_positions]
        self.underline_y_line = -0.055

        self.logger.log("Stimuli Temporal Judgement chargés.")

    def _setup_feedback_stimuli(self):
        self._line_user = visual.Line(
            self.win,
            start=(-0.04, self.underline_y_line),
            end=(0.04, self.underline_y_line),
            lineColor='yellow',
            lineWidth=5 * self.pixel_scale
        )
        self._line_correct = visual.Line(
            self.win,
            start=(-0.04, self.underline_y_line),
            end=(0.04, self.underline_y_line),
            lineColor='red',
            lineWidth=6 * self.pixel_scale
        )
        self._fb_text = visual.TextStim(
            self.win, text="", color='white', height=0.05, pos=(0, -0.2)
        )
        self._timeout_text = visual.TextStim(
            self.win, text="Temps de réponse écoulé", color='red', height=0.1
        )

    def log_trial_event(self, event_type, **kwargs):
        current_time = self.task_clock.getTime()

        if self.eyetracker_actif:
            self.EyeTracker.send_message(
                f"PHASE_{self.current_phase.upper()}_TRIAL_{(self.current_trial_idx if self.current_trial_idx is not None else 0):03d}_{event_type.upper()}"
            )

        entry = {
            'participant': self.nom,
            'session': self.session,
            'phase': self.current_phase,
            'trial': self.current_trial_idx,
            'time_s': current_time,
            'event_type': event_type
        }
        entry.update(kwargs)
        self.global_records.append(entry)

    def draw_lightbulb(self, base_color, bulb_on=False):
        self.colored_bar.fillColor = base_color
        self.colored_bar.lineColor = base_color
        self.colored_bar.draw()

        bulb = self.bulb_on_img if bulb_on else self.bulb_off_img
        bulb.draw()

    def _flip_with_onflip_events(self, trigger_code=None, log_event_type=None, log_kwargs=None):
        if trigger_code is not None and self.parport_actif:
            self.win.callOnFlip(self.ParPort.send_trigger, trigger_code)
        if log_event_type is not None:
            if log_kwargs is None:
                log_kwargs = {}
            self.win.callOnFlip(self.log_trial_event, log_event_type, **log_kwargs)
        return self.win.flip()

    def _set_line_at_index(self, line_stim, idx):
        x = self.underline_x_positions[idx]
        line_stim.start = (x - 0.04, self.underline_y_line)
        line_stim.end = (x + 0.04, self.underline_y_line)

    def _busy_wait_until(self, target_time_s, margin_s=None):
        if margin_s is None:
            margin_s = self.frame_tolerance_s
        now = self.task_clock.getTime()
        remaining = target_time_s - now
        if remaining <= 0:
            return
        coarse = remaining - max(0.0, 2.0 * self.frame_duration_s)
        if coarse > 0:
            core.wait(coarse)
        while self.task_clock.getTime() < (target_time_s - margin_s):
            core.wait(0.0)

    def run_trial(self, trial_index, total_trials, condition, delay_ms, feedback=False):
        should_quit(self.win)

        gc.disable()
        self._gc_counter += 1

        self.current_trial_idx = trial_index
        base_color = '#00FF00' if condition == 'active' else '#FF0000'

        self.log_trial_event('trial_start', condition=condition, delay_target_ms=delay_ms, feedback_mode=feedback)

        trigger_code = self.codes['trial_active'] if condition == 'active' else self.codes['trial_passive']

        self.fixation.draw()
        t_fix = self._flip_with_onflip_events(
            trigger_code=trigger_code,
            log_event_type='fixation_onset',
            log_kwargs={'condition': condition}
        )
        core.wait(0.5)

        self.draw_lightbulb(base_color=base_color, bulb_on=False)
        t_bulb_off = self._flip_with_onflip_events(log_event_type='bulb_off_onset')

        action_time = None
        action_source = None
        action_key = None

        if condition == 'active':
            while True:
                k = self._kb_get_keys(keyList=[self.key_action] + self.keys_quit)
                if k is None:
                    core.wait(0.0)
                    continue
                key_name, _, t_down = k
                if key_name in self.keys_quit:
                    should_quit(self.win, quit=True)
                action_time = t_down
                action_source = 'key'
                action_key = key_name
                if self.parport_actif:
                    self.ParPort.send_trigger(self.codes['action_bulb'])
                break
        else:
            jitter = random.uniform(*self.passive_action_jitter_s)
            action_time = self.task_clock.getTime() + jitter
            self._busy_wait_until(action_time, margin_s=self.frame_tolerance_s)
            action_source = 'passive_timer'
            action_key = None
            if self.parport_actif:
                self.ParPort.send_trigger(self.codes['action_bulb'])

        self.log_trial_event('action_performed', action_key=action_key, action_source=action_source, action_time_s=action_time)

        target_light_time = action_time + (delay_ms / 1000.0)
        self._busy_wait_until(target_light_time, margin_s=self.frame_tolerance_s)

        self.draw_lightbulb(base_color=base_color, bulb_on=True)
        t_bulb_on = self._flip_with_onflip_events(
            trigger_code=self.codes['bulb_on'],
            log_event_type='bulb_onset'
        )

        actual_delay = (t_bulb_on - action_time) * 1000.0
        error_ms = actual_delay - float(delay_ms)

        self.log_trial_event(
            'bulb_lit',
            bulb_on_time_s=t_bulb_on,
            bulb_off_onset_s=t_bulb_off,
            fixation_onset_s=t_fix,
            actual_delay_ms=actual_delay,
            error_ms=error_ms
        )

        core.wait(random.uniform(1.2, 1.8))
        self.win.flip()

        self.response_title.draw()
        self.response_options_text.draw()
        self.response_instr.draw()
        t_resp_screen = self._flip_with_onflip_events(
            trigger_code=self.codes['response_prompt'],
            log_event_type='response_prompt_onset'
        )

        resp = self._kb_wait_key(
            keyList=self.keys_responses + self.keys_quit,
            maxWait=self.response_deadline_s
        )

        rt = None
        response_ms = None

        if resp is not None:
            resp_key, _, t_down = resp

            if resp_key in self.keys_quit:
                should_quit(self.win, quit=True)

            if self.parport_actif:
                self.ParPort.send_trigger(self.codes['response_given'])

            rt = t_down - t_resp_screen
            response_ms = self.response_key_to_ms.get(resp_key)
            idx_user = self.keys_responses.index(resp_key)

            if feedback:
                is_correct = (response_ms == delay_ms)
                user_color = 'green' if is_correct else 'yellow'
                msg_text = "Bonne réponse !" if is_correct else f"Réponse correcte : {delay_ms} ms"
                msg_color = 'green' if is_correct else 'red'

                self._line_user.lineColor = user_color
                self._set_line_at_index(self._line_user, idx_user)

                self.response_title.draw()
                self.response_options_text.draw()
                self._line_user.draw()

                if not is_correct:
                    try:
                        idx_correct = self.response_values_ms.index(delay_ms)
                        self._set_line_at_index(self._line_correct, idx_correct)
                        self._line_correct.draw()
                    except ValueError:
                        pass

                self._fb_text.text = msg_text
                self._fb_text.color = msg_color
                self._fb_text.draw()
                self.win.flip()
                core.wait(1.0)
            else:
                self._line_user.lineColor = 'yellow'
                self._set_line_at_index(self._line_user, idx_user)

                self.response_title.draw()
                self.response_options_text.draw()
                self.response_instr.draw()
                self._line_user.draw()
                self.win.flip()
                core.wait(0.6)

            self.log_trial_event(
                'response_given',
                response_key=resp_key,
                response_ms=response_ms,
                rt_s=rt,
                response_time_s=t_down,
                response_screen_onset_s=t_resp_screen
            )

            fb_str = f"| FB: {'Yes' if feedback else 'No':<3}"
            self.logger.log(
                f"Trial {trial_index:>2}/{total_trials:<2} | {condition.upper():<7} | "
                f"Target: {delay_ms:>3}ms | Answer: {str(response_ms):>4}ms | RT: {rt:.3f}s {fb_str}"
            )

        else:
            if self.parport_actif:
                self.ParPort.send_trigger(self.codes['timeout'])
            self.log_trial_event('response_timeout', response_screen_onset_s=t_resp_screen)

            self._timeout_text.draw()
            self.win.flip()
            core.wait(0.8)

            self.logger.warn(
                f"Trial {trial_index:>2}/{total_trials:<2} | {condition.upper():<7} | "
                f"Target: {delay_ms:>3}ms | TIMEOUT"
            )

        if self._gc_counter % self.gc_collect_every == 0:
            gc.enable()
            gc.collect()
            gc.disable()

        gc.enable()

        isi = random.uniform(*self.stim_isi_range)
        self.fixation.draw()
        t_iti = self._flip_with_onflip_events(log_event_type='iti_onset')
        core.wait(isi)

        self.log_trial_event('trial_end', isi_duration=isi, iti_onset_s=t_iti)

        return True

    def build_trials(self, n_trials, training=False):
        conditions = ['active'] if training else ['active', 'passive']

        if training:
            unique_types = [(c, d) for c in conditions for d in self.delays_ms]
            n_full_repeats = n_trials // len(unique_types)
            remainder = n_trials % len(unique_types)
            trials = unique_types * n_full_repeats
            if remainder > 0:
                trials.extend(random.sample(unique_types, remainder))
            random.shuffle(trials)
            return trials

        def build_pool():
            n_per_condition = n_trials // 2
            remainder = n_trials % 2

            pool = []
            for cond in conditions:
                n_for_cond = n_per_condition + (1 if cond == 'active' and remainder else 0)
                n_per_delay = n_for_cond // len(self.delays_ms)
                delay_remainder = n_for_cond % len(self.delays_ms)

                for delay in self.delays_ms:
                    count = n_per_delay + (1 if delay_remainder > 0 else 0)
                    if delay_remainder > 0:
                        delay_remainder -= 1
                    pool.extend([(cond, delay)] * count)

            return pool

        def count_recent(trial_list, condition=None, delay=None):
            if not trial_list:
                return 0
            count = 0
            for trial in reversed(trial_list):
                if condition is not None and trial[0] == condition:
                    count += 1
                elif delay is not None and trial[1] == delay:
                    count += 1
                else:
                    break
            return count

        def is_valid(trial_list, candidate):
            cond, delay = candidate
            if count_recent(trial_list, condition=cond) >= 2:
                return False
            if count_recent(trial_list, delay=delay) >= 1:
                return False
            return True

        def get_best_candidates(pool, trial_list):
            if not trial_list:
                return pool[:]

            valid = [t for t in pool if is_valid(trial_list, t)]
            if not valid:
                return []

            def score_candidate(candidate):
                cond, delay = candidate
                cond_penalty = count_recent(trial_list, condition=cond) * 10
                delay_penalty = count_recent(trial_list, delay=delay) * 20
                diversity_bonus = 0
                if trial_list and trial_list[-1][0] != cond:
                    diversity_bonus += 5
                if trial_list and trial_list[-1][1] != delay:
                    diversity_bonus += 3
                return -(cond_penalty + delay_penalty - diversity_bonus)

            valid.sort(key=score_candidate, reverse=True)
            return valid

        def build_sequence(pool, max_attempts=50):
            for _ in range(max_attempts):
                sequence = []
                remaining = pool[:]
                random.shuffle(remaining)

                while len(sequence) < n_trials:
                    candidates = get_best_candidates(remaining, sequence)

                    if not candidates:
                        if len(sequence) < 3:
                            break
                        for _ in range(min(3, len(sequence))):
                            removed = sequence.pop()
                            remaining.append(removed)
                        random.shuffle(remaining)
                        continue

                    top_n = min(5, len(candidates))
                    chosen = random.choice(candidates[:top_n])

                    sequence.append(chosen)
                    remaining.remove(chosen)

                if len(sequence) == n_trials:
                    return sequence

            return None

        max_restarts = 20
        for _ in range(max_restarts):
            pool = build_pool()
            result = build_sequence(pool)
            if result:
                return result

        self.logger.warning(
            f"Contraintes difficiles à respecter pour {n_trials} trials. "
            "Utilisation d'une randomisation simple."
        )
        pool = build_pool()
        random.shuffle(pool)
        return pool

    def run_trial_block(self, n_trials, block_name, phase_tag, feedback):
        self.current_phase = phase_tag
        self.log_trial_event('block_start', block_name=block_name, feedback_mode=feedback)
        self.logger.log(f"--- Bloc Start: {block_name} ({n_trials} essais) ---")

        trials = self.build_trials(n_trials, training=(phase_tag == 'training'))
        total_trials = len(trials)

        for i, (cond, delay) in enumerate(trials, start=1):
            self.run_trial(i, total_trials, cond, delay, feedback=feedback)

        self.log_trial_event('block_end', block_name=block_name)
        self.logger.log(f"--- Bloc End: {block_name} ---")

    def show_crisis_validation_window(self):
        self.current_phase = 'crisis_validation'
        self.current_trial_idx = None
        self.logger.log("=== Entering Crisis Validation Window ===")

        loop_crisis = True

        while loop_crisis:
            msg_launch = visual.TextStim(
                self.win, text='Appuyez pour démarrer la crise', height=0.08
            )
            msg_launch.draw()
            self.win.flip()

            self.log_trial_event('crisis_prompt_start')
            if self.parport_actif:
                self.ParPort.send_trigger(self.codes['crisis_prompt'])

            k = self._kb_wait_key(keyList=self.keys_responses + self.keys_quit, maxWait=None)
            if k is None:
                continue
            if k[0] in self.keys_quit:
                should_quit(self.win, quit=True)

            self.log_trial_event('crisis_action_started', trigger_key=k[0])
            if self.parport_actif:
                self.ParPort.send_trigger(self.codes['crisis_start'])

            self.fixation.draw()
            self.win.flip()
            core.wait(0.5)

            k = self._kb_wait_key(keyList=self.keys_responses + self.keys_quit, maxWait=None)
            if k is None:
                continue
            if k[0] in self.keys_quit:
                should_quit(self.win, quit=True)

            if self.parport_actif:
                self.ParPort.send_trigger(self.codes['crisis_end'])
            self.log_trial_event('crisis_action_ended', end_key=k[0])

            choice_text = visual.TextStim(
                self.win,
                text='[1-4] Crise réussie     [5-8] Crise échouée',
                height=0.08
            )
            choice_text.draw()
            self.win.flip()

            self.log_trial_event('crisis_validation_prompt')
            if self.parport_actif:
                self.ParPort.send_trigger(self.codes['crisis_valid_prompt'])

            core.wait(0.2)

            k = self._kb_wait_key(keyList=self.keys_responses + self.keys_quit, maxWait=None)
            if k is None:
                continue
            if k[0] in self.keys_quit:
                should_quit(self.win, quit=True)

            key = k[0]
            idx = self.keys_responses.index(key)
            success = True if idx < 4 else False
            result_label = 'SUCCESS' if success else 'FAILED'

            trigger_code = self.codes['crisis_res_success'] if success else self.codes['crisis_res_fail']
            if self.parport_actif:
                self.ParPort.send_trigger(trigger_code)

            self.log_trial_event('crisis_result_chosen', result=result_label, key=key)
            self.logger.log(f"Crisis Outcome: {result_label}")

            confirmation = visual.TextStim(
                self.win, text=f"Résultat : {result_label}", height=0.08
            )
            confirmation.draw()
            self.win.flip()
            core.wait(1.0)

            if not success:
                retry_text = visual.TextStim(
                    self.win,
                    text='Recommencer ?\n[1-4] Oui   [5-8] Non (Quitter)',
                    height=0.06
                )
                retry_text.draw()
                self.win.flip()

                k = self._kb_wait_key(keyList=self.keys_responses + self.keys_quit, maxWait=None)
                if k is None:
                    continue
                if k[0] in self.keys_quit:
                    should_quit(self.win, quit=True)

                idx_retry = self.keys_responses.index(k[0])

                if idx_retry >= 4:
                    self.log_trial_event('crisis_retry_decision', choice='no_retry_quit')
                    if self.parport_actif:
                        self.ParPort.send_trigger(self.codes['crisis_retry_no'])
                    should_quit(self.win, quit=True)
                else:
                    self.log_trial_event('crisis_retry_decision', choice='retry')
                    if self.parport_actif:
                        self.ParPort.send_trigger(self.codes['crisis_retry_yes'])
                    self.logger.log("Crisis Retry Selected.")
            else:
                loop_crisis = False

        self.log_trial_event('crisis_phase_end')

    def show_instructions(self, text_override=None):
        msg = text_override if text_override else f"Bienvenue dans la tâche : {self.task_name}\n\nAppuyez sur une touche pour voir les consignes spécifiques."
        self.instr_stim.text = msg
        self.instr_stim.draw()
        self.win.flip()
        core.wait(0.5)
        self._kb_wait_key(keyList=None, maxWait=None)

    def wait_for_trigger(self, trigger_key='t'):
        self.instr_stim.text = "En attente du trigger IRM..."
        self.instr_stim.draw()
        self.win.flip()

        self.logger.log("Waiting for trigger...")

        k = self._kb_wait_key(keyList=[trigger_key], maxWait=None)
        while k is None:
            k = self._kb_wait_key(keyList=[trigger_key], maxWait=None)

        self.task_clock.reset()

        start_code = self.codes.get('start_exp', 255)
        if self.parport_actif:
            self.ParPort.send_trigger(start_code)

        if self.eyetracker_actif:
            self.EyeTracker.start_recording()
            self.EyeTracker.send_message(f"START_{self.task_name.upper()}")

        self.logger.log(f"Trigger reçu. Start Code: {start_code}")

    def run(self):
        finished_naturally = False

        try:
            if self.run_type == 'training':
                instructions = (
                    "ENTRAINEMENT - Tâche de Jugement Temporel\n\n"
                    "Vous allez voir une ampoule.\n"
                    "Condition ACTIVE (barre verte) : Appuyez sur le bouton pour l'allumer.\n"
                    "Condition PASSIVE (barre rouge) : Elle s'allumera automatiquement.\n\n"
                    "Après un délai, estimez le temps perçu (100 à 800ms).\n\n"
                    f"Nombre d'essais : {self.n_trials_training} (avec feedback)\n\n"
                    "Appuyez sur 't' (trigger) pour commencer..."
                )
            else:
                instructions = (
                    "Tâche de Jugement Temporel\n\n"
                    "Condition ACTIVE (barre verte) : Appuyez pour allumer l'ampoule.\n"
                    "Condition PASSIVE (barre rouge) : Attente passive.\n\n"
                    "Ensuite, évaluez le délai perçu (100 à 800ms).\n\n"
                    "Appuyez sur une touche pour continuer..."
                )

            self.show_instructions(instructions)

            self.wait_for_trigger(trigger_key=self.key_trigger)

            if self.run_type == 'training':
                self.logger.log(f"Lancement : TRAINING ({self.n_trials_training} essais)")
                self.show_resting_state(duration_s=10.0)
                self.run_trial_block(
                    self.n_trials_training,
                    block_name="TRAINING",
                    phase_tag='training',
                    feedback=True
                )

            elif self.run_type == 'base':
                self.logger.log("Lancement : PROTOCOLE COMPLET")
                self.show_resting_state(duration_s=10.0)
                self.run_trial_block(
                    self.n_trials_base,
                    block_name="BASELINE",
                    phase_tag='base',
                    feedback=False
                )
                self.show_crisis_validation_window()
                self.run_trial_block(
                    self.n_trials_block,
                    block_name="POST_CRISIS",
                    phase_tag='run_standard',
                    feedback=False
                )

            else:
                self.logger.log("Lancement : BLOC COURT")
                self.show_resting_state(duration_s=150.0)
                self.show_crisis_validation_window()
                self.run_trial_block(
                    self.n_trials_block,
                    block_name="STANDARD_BLOCK",
                    phase_tag='run_standard',
                    feedback=False
                )

            finished_naturally = True
            self.logger.ok("Expérience terminée avec succès.")

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Interruption manuelle.")

        except Exception as e:
            self.logger.err(f"ERREUR CRITIQUE : {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            self.logger.log("Sauvegarde finale...")

            if self.eyetracker_actif:
                self.EyeTracker.stop_recording()
                self.EyeTracker.send_message("END_EXP")
                self.EyeTracker.close_and_transfer_data(self.data_dir)

            self.save_data(
                data_list=self.global_records,
                filename_suffix=f"_{self.run_type}"
            )

            list_of_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                try:
                    qc_temporaljudgement(latest_file)
                except Exception as e:
                    self.logger.warn(f"Echec génération QC: {e}")

            if finished_naturally:
                end_msg = "Fin de la session.\nMerci pour votre participation."
                self.show_instructions(end_msg)
                core.wait(3.0)