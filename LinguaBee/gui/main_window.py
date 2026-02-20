import os
import time
import re

# UI / Qt imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QProgressBar, QTextEdit, QMessageBox, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QTextCursor

# Local imports
from config import TranscriptionConfig, downloads_dir, icons_dir, save_settings, load_settings
from worker import TranscriptionThread
from huggingface_utils import get_hf_token, save_hf_token
from gui.widgets import (
    SourceGroup, TranscriptionGroup, TranslationGroup, SummaryGroup, FormatsGroup
)
from gui.dialogs import (
    WhisperSettingsDialog, NllbSettingsDialog, OllamaSettingsDialog, BartSummarizationSettingsDialog, DiarizationDialog,
    CorrectionSettingsDialog, GeminiCorrectionSettingsDialog, OpenRouterCorrectionSettingsDialog,
    OllamaSummarySettingsDialog, GeminiSummarySettingsDialog, OpenRouterSummarySettingsDialog,
    OpenRouterTranslationSettingsDialog, OllamaTranslationSettingsDialog
)

# torch / CUDA detection
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LinguaBee")

        self.local_files = []
        # Populate available devices (CPU + CUDA GPUs when available)
        self.available_devices = self._detect_available_devices()
        self.whisper_variant = "medium"
        self.whisper_device = "cpu"
        self.whisper_device_index = 0
        self.whisper_delete_audio = False
        self.whisper_enable_paragraphing = True
        self.whisper_enable_denoising = False
        self.whisper_enable_normalization = False
        self.whisper_force_mono = True
        self.transcription_segment_batch_size = 250
        self.nllb_variant = None
        self.nllb_device = "cpu"
        self.nllb_device_index = 0
        self.translation_segment_batch_size = 250
        self.hf_summary_model_name = "mtj/bart-base-polish-summarization"
        self.hf_summary_device = "cpu"
        self.hf_summary_device_index = 0
        self.hf_summary_max_length = 150
        self.hf_summary_min_length = 30
        self.hf_summary_num_beams = 4
        self.ollama_model_name = ""
        self.ollama_translation_prompt = ""
        self.translation_openrouter_model_name = "google/gemini-2.5-flash"
        self.translation_openrouter_prompt = ""
        self.ollama_summary_model_name = ""
        self.ollama_summary_prompt = ""
        self.bart_summary_prompt = ""
        self.log_level = "info"
        # Correction (Korekta) settings
        self.correction_ollama_model_name = ""
        self.correction_prompt = ""
        self.gemini_key = ""
        self.openrouter_key = ""
        self.openrouter_model_name = "google/gemini-2.5-flash"
        self.summary_gemini_prompt = ""
        self.summary_openrouter_prompt = ""
        self.summary_openrouter_model_name = "google/gemini-2.5-flash"
        # Forced-alignment settings
        self.enable_forced_alignment = False
        self.forced_alignment_model = ""
        
        # Diarization settings (now part of Whisper settings)
        self.diarization_hf_token = get_hf_token() or ""
        self.diarization_num_speakers = 0
        self.enable_diarization = False

        central = QWidget()
        self.setCentralWidget(central)

        # Build UI
        self._init_widgets()
        self._init_layout()
        self._connect_signals()

        # Load persisted settings from previous sessions (if any) and apply
        try:
            settings = load_settings()
            # Prevent immediate re-saving while applying
            self._suppress_save = True
            self._apply_settings(settings)
            self._suppress_save = False
        except Exception:
            try:
                self._suppress_save = False
            except Exception:
                pass

        # Environment info logging (best-effort)
        try:
            self._log_environment_info()
        except Exception:
            pass

    def _detect_available_devices(self):
        """Return a list of (display_name, data_dict) for available devices.

        data_dict contains at least 'device' and optionally 'device_index'.
        """
        devices = []
        # Always offer CPU
        devices.append(("CPU", {'device': 'cpu'}))
        try:
            if TORCH_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        count = torch.cuda.device_count()
                        for i in range(count):
                            try:
                                name = torch.cuda.get_device_name(i)
                            except Exception:
                                name = f"cuda:{i}"
                            disp = f"GPU {i}: {name} (cuda:{i})"
                            devices.append((disp, {'device': 'cuda', 'device_index': i}))
                except Exception:
                    # If checking CUDA fails, skip GPU entries
                    pass
        except Exception:
            pass
        return devices

    def _init_widgets(self):
        self.source_group = SourceGroup()
        # Adjust the visual height of the source selection group: previously set to ~50%,
        # increase by 50% of that value so final factor is ~0.75 of the original sizeHint.
        try:
            hint_h = self.source_group.sizeHint().height()
            # previously used ~75% of hint; increase that by 30% (final ~97.5% of original)
            base = int(hint_h * 0.75)
            final_h = max(90, int(base * 1.3))
            self.source_group.setFixedHeight(final_h)
        except Exception:
            # If sizeHint isn't usable yet, fall back to a slightly larger default
            try:
                self.source_group.setFixedHeight(160)
            except Exception:
                pass
        self.transcription_group = TranscriptionGroup()
        self.translation_group = TranslationGroup()
        self.summary_group = SummaryGroup()
        self.formats_group = FormatsGroup()
        
        self.progress_bar = QProgressBar()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.current_eta = "--:--"
        
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("StartButton")
        self.stop_btn = QPushButton("PRZERWIJ")
        self.stop_btn.setObjectName("StopButton")
        self.stop_btn.setEnabled(False)
        from PySide6.QtWidgets import QLabel



    # place log level control near the log box in layout init

    def _init_layout(self):
        self.layout = QVBoxLayout(self.centralWidget())
        
        logo_label = QLabel()
        # Load source pixmap
        src_pix = QPixmap(os.path.join(icons_dir, 'LinguaBee_512x512_navy.png'))
        # Aim to align the logo height with the visual frame height of the SourceGroup (top/bottom lines)
        # sizeHint may be small before layout; estimate by accounting for groupbox padding and title height
        hint = self.source_group.sizeHint().height()
        title_height = self.source_group.fontMetrics().height() + 8
        groupbox_padding = 15  # matches stylesheet QGroupBox padding
        # Estimate the visual frame height: sizeHint plus padding + title area
        target_h = hint + (2 * groupbox_padding) + title_height
        # If sizeHint is unusable, fallback to a larger default so it fills the group frame
        if not target_h or target_h < 140:
            target_h = 160
        # Make logo height match the visible height of the SourceGroup so top/bottom edges align
        try:
            sg_h = int(self.source_group.height())
            if sg_h and sg_h > 20:
                target_h = sg_h
            else:
                target_h = max(80, int(target_h * 0.75))
        except Exception:
            target_h = target_h
        # Scale while preserving aspect ratio
        scaled_pix = src_pix.scaledToHeight(target_h, Qt.SmoothTransformation)

        # Use the same corner radius as QGroupBox in stylesheet (8px) so rounding matches
        radius = 8
        rounded = QPixmap(scaled_pix.size())
        rounded.fill(Qt.transparent)
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(rounded.rect(), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, scaled_pix)
        painter.end()

        logo_label.setPixmap(rounded)
        # Fix the widget size to the pixmap so it lines up with the SourceGroup frame
        logo_label.setFixedSize(rounded.size())
        logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        main_grid = QGridLayout()
        main_grid.addWidget(self.source_group, 0, 0, 1, 2)
        main_grid.addWidget(logo_label, 0, 2, Qt.AlignCenter)
        main_grid.addWidget(self.transcription_group, 1, 0)
        main_grid.addWidget(self.translation_group, 1, 1)
        main_grid.addWidget(self.summary_group, 1, 2)
        main_grid.addWidget(self.formats_group, 2, 0, 1, 3)

        self.layout.addLayout(main_grid)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.log_box)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_btn, 3)
        button_layout.addWidget(self.stop_btn, 1)
        self.layout.addLayout(button_layout)

    def _connect_signals(self):
        self.source_group.url_entry.textChanged.connect(self.clear_local_files)
        self.source_group.file_btn.clicked.connect(self.choose_file)
        self.start_btn.clicked.connect(self.start_transcription)
        self.stop_btn.clicked.connect(self.stop_transcription)

        self.transcription_group.model_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.transcription_group.src_lang_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.transcription_group.correction_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.transcription_group.model_combo.activated.connect(self.open_transcription_settings)
        # Open correction settings when user selects Ollama (lokalny) in Korekta
        self.transcription_group.correction_combo.activated.connect(self.open_correction_settings)

        self.translation_group.translation_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.translation_group.translation_src_lang_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.translation_group.tgt_lang_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.translation_group.translation_combo.activated.connect(self.open_translation_settings)

        self.summary_group.summary_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.summary_group.summary_lang_combo.currentTextChanged.connect(self._on_model_or_language_changed)
        self.summary_group.summary_combo.activated.connect(self.open_summary_settings)
        
        # Save settings whenever format checkboxes change
        for cb in getattr(self.formats_group, 'original_checkboxes', []):
            cb.stateChanged.connect(self._save_current_settings)
        for cb in getattr(self.formats_group, 'translated_checkboxes', []):
            cb.stateChanged.connect(self._save_current_settings)
        for cb in getattr(self.formats_group, 'summary_checkboxes', []):
            cb.stateChanged.connect(self._save_current_settings)

    @Slot(str)
    def _on_model_or_language_changed(self, *_args):
        self._update_ui_state()
        self._save_current_settings()


    @Slot()
    def clear_local_files(self):
        self.local_files = []
        self.source_group.file_btn.setText("WYBIERZ PLIK LOKALNY")
        self.transcription_group.model_combo.setEnabled(True)

    @Slot()
    def _update_ui_state(self):
        transcription_model = self.transcription_group.model_combo.currentText()
        supports_text_pipeline = transcription_model in ("Whisper (lokalny)", "Brak")
        is_local_whisper = transcription_model == "Whisper (lokalny)"
        self.transcription_group.src_lang_combo.setEnabled(supports_text_pipeline)
        # Enable/disable correction option when using local Whisper
        try:
            self.transcription_group.correction_combo.setEnabled(supports_text_pipeline)
        except Exception:
            pass
        self.translation_group.setEnabled(supports_text_pipeline)
        self.summary_group.setEnabled(supports_text_pipeline)

        if not supports_text_pipeline:
            self.translation_group.translation_src_lang_combo.setEnabled(False)
            self.translation_group.tgt_lang_combo.setEnabled(False)
        else:
            selected_translation_model = self.translation_group.translation_combo.currentText()
            is_any_translation_selected = selected_translation_model != "Brak"
            self.translation_group.translation_src_lang_combo.setEnabled(is_any_translation_selected)
            self.translation_group.tgt_lang_combo.setEnabled(is_any_translation_selected)


    @Slot(str, str)
    def append_log(self, msg, msg_type="info"):
        current_time = time.strftime("%H:%M:%S")
        color_map = {"error": "red", "warning": "yellow", "success": "green", "info": "white"}
        color = color_map.get(msg_type, "white")
        # filter by selected log level
        order = {"debug": 10, "info": 20, "warning": 30, "error": 40}
        if order.get(msg_type, 20) < order.get(self.log_level, 20):
            return
        # Insert HTML at the end of the QTextEdit using a QTextCursor
        try:
            try:
                m = re.search(r"ETA:\s*([^\s]+)", str(msg))
                if m:
                    self.current_eta = m.group(1).strip()
                    self.progress_bar.setFormat(f"%p% | ETA: {self.current_eta}")
            except Exception:
                pass
            html = f'<span style="color:#888;">[{current_time}] </span><span style="color:{color};">{msg}</span><br/>'
            cursor = self.log_box.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertHtml(html)
            self.log_box.setTextCursor(cursor)
            # ensure the newly added text is visible
            try:
                self.log_box.ensureCursorVisible()
            except Exception:
                pass
        except Exception:
            # Fallback: append plain text
            self.log_box.append(f'[{current_time}] {msg}')

    def choose_file(self):
        file_filter = (
            "Wszystkie wspierane (*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.mp4 *.mkv *.mov *.avi *.webm *.srt *.txt *.docx *.html *.htm);;"
            "Pliki tekstowe (*.txt *.docx *.html *.htm *.srt);;"
            "Pliki audio/wideo (*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.mp4 *.mkv *.mov *.avi *.webm);;"
            "Wszystkie pliki (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(self, "Wybierz pliki", "", file_filter)
        if paths:
            self.local_files = paths
            self.source_group.file_btn.setText(
                f"WYBRANO {len(paths)} PLIKÓW" if len(paths) > 1 else os.path.basename(paths[0])
            )
            text_like_exts = {'.txt', '.docx', '.html', '.htm', '.srt'}
            has_text_file = any(os.path.splitext(p)[1].lower() in text_like_exts for p in paths)
            if has_text_file:
                self.transcription_group.model_combo.setCurrentText("Brak")
                self.transcription_group.model_combo.setEnabled(False)
                self.append_log("Wykryto plik tekstowy: model transkrypcji ustawiono na 'Brak'.", "info")
            else:
                self.transcription_group.model_combo.setEnabled(True)
            # Detailed log about chosen files
            if len(paths) > 1:
                self.append_log(f"Wybrano {len(paths)} plików: {', '.join([os.path.basename(p) for p in paths])}", "info")
            else:
                self.append_log(f"Wybrano plik: {os.path.basename(paths[0])}", "info")
            self.source_group.url_entry.clear()
            # also log file sizes for selected files
            try:
                total_bytes = 0
                for p in paths:
                    try:
                        sz = os.path.getsize(p)
                        total_bytes += sz
                        self.append_log(f"Plik: {os.path.basename(p)}, rozmiar: {sz} bytes", "debug")
                    except Exception:
                        self.append_log(f"Nie można odczytać rozmiaru pliku: {p}", "warning")
                self.append_log(f"Łączny rozmiar wybranych plików: {total_bytes} bytes", "info")
            except Exception:
                pass

    def stop_transcription(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.append_log("Próba zatrzymania procesu...", "warning")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def open_transcription_settings(self, index):
        model = self.transcription_group.model_combo.itemText(index)
        if model == "Whisper (lokalny)":
            dialog = WhisperSettingsDialog(
                self, 
                self.available_devices, 
                self.whisper_variant, 
                self.whisper_device, 
                self.whisper_device_index, 
                self.whisper_delete_audio, 
                self.whisper_enable_paragraphing,
                self.whisper_enable_denoising,
                self.whisper_enable_normalization,
                self.whisper_force_mono,
                self.enable_diarization,
                self.diarization_num_speakers,
                self.transcription_segment_batch_size,
            )
            if dialog.exec():
                settings = dialog.get_settings()
                (self.whisper_variant, device_settings, self.whisper_delete_audio, 
                 self.whisper_enable_paragraphing, self.whisper_enable_denoising, self.whisper_enable_normalization, 
                 self.whisper_force_mono, self.enable_diarization, self.diarization_num_speakers,
                 self.transcription_segment_batch_size) = settings
                self.whisper_device = device_settings['device']
                self.whisper_device_index = device_settings.get('device_index', 0)
                self.append_log(f"Ustawiono Whisper: {self.whisper_variant}, {self.whisper_device}", "info")
                # persist
                self._save_current_settings()

    def open_diarization_settings(self):
        dialog = DiarizationDialog(
            self,
            current_hf_token=self.diarization_hf_token,
            current_num_speakers=self.diarization_num_speakers
        )
        if dialog.exec():
            settings = dialog.get_settings()
            self.diarization_hf_token = settings["hf_token"]
            self.diarization_num_speakers = settings["num_speakers"]
            if self.diarization_hf_token:
                save_hf_token(self.diarization_hf_token)
            self.append_log(f"Zapisano ustawienia diaryzacji: {self.diarization_num_speakers} mówców.", "info")
            self._save_current_settings()

    def open_translation_settings(self, index):
        model = self.translation_group.translation_combo.itemText(index)
        if model == "NLLB (lokalny)":
            dialog = NllbSettingsDialog(
                self,
                self.available_devices,
                self.nllb_variant,
                self.nllb_device,
                self.nllb_device_index,
                self.translation_segment_batch_size,
            )
            if dialog.exec():
                self.nllb_variant, device_settings, self.translation_segment_batch_size = dialog.get_settings()
                self.nllb_device = device_settings['device']
                self.nllb_device_index = device_settings.get('device_index', 0)
                self.append_log(f"Ustawiono NLLB: {self.nllb_variant}, {self.nllb_device}", "info")
                # persist
                self._save_current_settings()
        elif model == "Ollama (lokalny)":
            dialog = OllamaTranslationSettingsDialog(
                self,
                current_model=self.ollama_model_name,
                current_prompt=self.ollama_translation_prompt,
                current_translation_segment_batch_size=self.translation_segment_batch_size,
            )
            if dialog.exec():
                self.ollama_model_name, self.ollama_translation_prompt, self.translation_segment_batch_size = dialog.get_settings()
                self.append_log(f"Ustawiono model Ollama dla tłumaczenia: {self.ollama_model_name}", "info")
                self._save_current_settings()
        elif model == "OpenRouter (API)":
            dialog = OpenRouterTranslationSettingsDialog(
                self,
                current_key=self.openrouter_key,
                current_prompt=self.translation_openrouter_prompt,
                current_model=self.translation_openrouter_model_name,
                current_translation_segment_batch_size=self.translation_segment_batch_size,
            )
            if dialog.exec():
                key, prompt, model_name, batch_size = dialog.get_settings()
                self.openrouter_key = key
                self.translation_openrouter_prompt = prompt
                self.translation_openrouter_model_name = model_name or "google/gemini-2.5-flash"
                self.translation_segment_batch_size = int(batch_size or self.translation_segment_batch_size)
                try:
                    st = load_settings() or {}
                    st['openrouter_key'] = self.openrouter_key
                    st['translation_openrouter_model_name'] = self.translation_openrouter_model_name
                    save_settings(st)
                except Exception:
                    pass
                self.append_log(f"Ustawiono model OpenRouter dla tłumaczenia: {self.translation_openrouter_model_name}", "info")
                self._save_current_settings()

    def open_correction_settings(self, index):
        model = self.transcription_group.correction_combo.itemText(index)
        if model == "Ollama (lokalny)":
            dialog = CorrectionSettingsDialog(self, current_model=self.correction_ollama_model_name, current_prompt=self.correction_prompt)
            if dialog.exec():
                model_name, prompt = dialog.get_settings()
                self.correction_ollama_model_name = model_name
                self.correction_prompt = prompt
                self.append_log(f"Ustawiono Korektę: model={model_name}", "info")
                self._save_current_settings()
        elif model == "Gemini (API)":
            dlg = GeminiCorrectionSettingsDialog(
                self,
                current_key=self.gemini_key,
                current_prompt=self.correction_prompt,
                current_transcription_segment_batch_size=self.transcription_segment_batch_size,
            )
            if dlg.exec():
                key, prompt, batch_size = dlg.get_settings()
                self.gemini_key = key
                self.correction_prompt = prompt
                self.transcription_segment_batch_size = int(batch_size or self.transcription_segment_batch_size)
                # persist
                try:
                    st = load_settings() or {}
                    st['gemini_key'] = self.gemini_key
                    save_settings(st)
                except Exception:
                    pass
                self.append_log("Zapisano ustawienia korekty Gemini.", "info")
                self._save_current_settings()
        elif model == "OpenRouter (API)":
            dlg = OpenRouterCorrectionSettingsDialog(
                self,
                current_key=self.openrouter_key,
                current_prompt=self.correction_prompt,
                current_model=self.openrouter_model_name,
                current_transcription_segment_batch_size=self.transcription_segment_batch_size,
            )
            if dlg.exec():
                key, prompt, model_name, batch_size = dlg.get_settings()
                self.openrouter_key = key
                self.correction_prompt = prompt
                self.openrouter_model_name = model_name or "google/gemini-2.5-flash"
                self.transcription_segment_batch_size = int(batch_size or self.transcription_segment_batch_size)
                try:
                    st = load_settings() or {}
                    st['openrouter_key'] = self.openrouter_key
                    st['openrouter_model_name'] = self.openrouter_model_name
                    save_settings(st)
                except Exception:
                    pass
                self.append_log(f"Zapisano ustawienia korekty OpenRouter (model: {self.openrouter_model_name}).", "info")
                self._save_current_settings()

    def open_summary_settings(self, index):
        model = self.summary_group.summary_combo.itemText(index)
        if model == "Ollama (lokalny)":
            dialog = OllamaSummarySettingsDialog(self, self.ollama_summary_model_name, self.ollama_summary_prompt)
            if dialog.exec():
                self.ollama_summary_model_name, self.ollama_summary_prompt = dialog.get_settings()
                self.append_log(f"Ustawiono model Ollama dla streszczenia: {self.ollama_summary_model_name}", "info")
                self._save_current_settings()
        elif model == "Gemini (API)":
            dlg = GeminiSummarySettingsDialog(self, current_key=self.gemini_key, current_prompt=self.summary_gemini_prompt)
            if dlg.exec():
                key, prompt = dlg.get_settings()
                self.gemini_key = key
                self.summary_gemini_prompt = prompt
                try:
                    st = load_settings() or {}
                    st['gemini_key'] = self.gemini_key
                    save_settings(st)
                except Exception:
                    pass
                self.append_log("Zapisano ustawienia streszczenia Gemini.", "info")
                self._save_current_settings()
        elif model == "OpenRouter (API)":
            dlg = OpenRouterSummarySettingsDialog(
                self,
                current_key=self.openrouter_key,
                current_prompt=self.summary_openrouter_prompt,
                current_model=self.summary_openrouter_model_name,
            )
            if dlg.exec():
                key, prompt, model_name = dlg.get_settings()
                self.openrouter_key = key
                self.summary_openrouter_prompt = prompt
                self.summary_openrouter_model_name = model_name or "google/gemini-2.5-flash"
                try:
                    st = load_settings() or {}
                    st['openrouter_key'] = self.openrouter_key
                    st['summary_openrouter_model_name'] = self.summary_openrouter_model_name
                    save_settings(st)
                except Exception:
                    pass
                self.append_log(f"Zapisano ustawienia streszczenia OpenRouter (model: {self.summary_openrouter_model_name}).", "info")
                self._save_current_settings()
        elif model == "BART (lokalny)":
            if not self.hf_summary_model_name or "bart" not in self.hf_summary_model_name.lower():
                self.hf_summary_model_name = "mtj/bart-base-polish-summarization"
            dialog = BartSummarizationSettingsDialog(
                self, 
                self.available_devices,
                self.hf_summary_model_name,
                self.hf_summary_device,
                self.hf_summary_device_index,
                self.hf_summary_max_length,
                self.hf_summary_min_length,
                self.hf_summary_num_beams,
                self.bart_summary_prompt,
            )
            if dialog.exec():
                settings = dialog.get_settings()
                self.hf_summary_model_name = settings["model_name"]
                self.hf_summary_device = settings["device"]
                self.hf_summary_device_index = settings["device_index"]
                self.hf_summary_max_length = settings["max_length"]
                self.hf_summary_min_length = settings["min_length"]
                self.hf_summary_num_beams = settings["num_beams"]
                self.bart_summary_prompt = settings.get("prompt", self.bart_summary_prompt)
                self.append_log(f"Ustawiono model podsumowujący HF: {self.hf_summary_model_name} na {self.hf_summary_device}", "info")
                self._save_current_settings()

    def _apply_settings(self, settings: dict):
        """Apply persisted settings to UI fields and internal variables."""
        try:
            self.whisper_variant = settings.get('whisper_variant', self.whisper_variant)
            allowed_whisper_variants = {"medium", "large-v3", "turbo"}
            if self.whisper_variant not in allowed_whisper_variants:
                self.whisper_variant = "medium"
            self.whisper_device = settings.get('whisper_device', self.whisper_device)
            self.whisper_device_index = settings.get('whisper_device_index', self.whisper_device_index)
            self.nllb_variant = settings.get('nllb_variant', self.nllb_variant)
            self.nllb_device = settings.get('nllb_device', self.nllb_device)
            self.nllb_device_index = settings.get('nllb_device_index', self.nllb_device_index)
            self.hf_summary_model_name = settings.get('hf_summary_model_name', self.hf_summary_model_name)
            self.hf_summary_device = settings.get('hf_summary_device', self.hf_summary_device)
            self.hf_summary_device_index = settings.get('hf_summary_device_index', self.hf_summary_device_index)
            self.hf_summary_max_length = settings.get('hf_summary_max_length', self.hf_summary_max_length)
            self.hf_summary_min_length = settings.get('hf_summary_min_length', self.hf_summary_min_length)
            self.hf_summary_num_beams = settings.get('hf_summary_num_beams', self.hf_summary_num_beams)
            self.bart_summary_prompt = settings.get('bart_summary_prompt', self.bart_summary_prompt)
            self.ollama_model_name = settings.get('ollama_model_name', self.ollama_model_name)
            self.ollama_translation_prompt = settings.get('ollama_translation_prompt', self.ollama_translation_prompt)
            self.translation_openrouter_model_name = settings.get('translation_openrouter_model_name', self.translation_openrouter_model_name)
            self.translation_openrouter_prompt = settings.get('translation_openrouter_prompt', self.translation_openrouter_prompt)
            self.ollama_summary_model_name = settings.get('ollama_summary_model_name', self.ollama_summary_model_name)
            self.ollama_summary_prompt = settings.get('ollama_summary_prompt', self.ollama_summary_prompt)
            self.summary_gemini_prompt = settings.get('summary_gemini_prompt', self.summary_gemini_prompt)
            self.summary_openrouter_prompt = settings.get('summary_openrouter_prompt', self.summary_openrouter_prompt)
            self.whisper_delete_audio = settings.get('whisper_delete_audio', self.whisper_delete_audio)
            self.whisper_enable_paragraphing = settings.get('whisper_enable_paragraphing', self.whisper_enable_paragraphing)
            self.whisper_enable_denoising = settings.get('enable_denoising', self.whisper_enable_denoising)
            self.whisper_enable_normalization = settings.get('enable_normalization', self.whisper_enable_normalization)
            self.whisper_force_mono = settings.get('force_mono', self.whisper_force_mono)
            self.transcription_segment_batch_size = settings.get('transcription_segment_batch_size', self.transcription_segment_batch_size)
            self.translation_segment_batch_size = settings.get('translation_segment_batch_size', self.translation_segment_batch_size)

            # Diarization
            self.diarization_num_speakers = settings.get('diarization_num_speakers', self.diarization_num_speakers)
            self.enable_diarization = settings.get('enable_diarization', self.enable_diarization)
            
            # Apply to UI combo boxes where possible
            try:
                self.transcription_group.model_combo.setCurrentText(settings.get('transcription_model', self.transcription_group.model_combo.currentText()))
            except Exception:
                pass
            try:
                self.transcription_group.src_lang_combo.setCurrentText(settings.get('transcription_src_language', self.transcription_group.src_lang_combo.currentText()))
            except Exception:
                pass
            try:
                self.transcription_group.correction_combo.setCurrentText(settings.get('transcription_correction', self.transcription_group.correction_combo.currentText()))
            except Exception:
                pass
            try:
                self.correction_ollama_model_name = settings.get('correction_ollama_model_name', self.correction_ollama_model_name)
                self.correction_prompt = settings.get('correction_prompt', self.correction_prompt)
                self.gemini_key = settings.get('gemini_key', self.gemini_key)
                self.openrouter_key = settings.get('openrouter_key', self.openrouter_key)
                self.openrouter_model_name = settings.get('openrouter_model_name', self.openrouter_model_name)
                self.summary_openrouter_model_name = settings.get('summary_openrouter_model_name', self.summary_openrouter_model_name)
                self.enable_forced_alignment = settings.get('enable_forced_alignment', self.enable_forced_alignment)
                self.forced_alignment_model = settings.get('forced_alignment_model', self.forced_alignment_model)
                # show_ollama_progress removed: progress now always shown when correction runs
                pass
            except Exception:
                pass
            try:
                self.translation_group.translation_combo.setCurrentText(settings.get('translation_model', self.translation_group.translation_combo.currentText()))
            except Exception:
                pass
            try:
                self.translation_group.translation_src_lang_combo.setCurrentText(settings.get('translation_src_language', self.translation_group.translation_src_lang_combo.currentText()))
            except Exception:
                pass
            try:
                self.translation_group.tgt_lang_combo.setCurrentText(settings.get('translation_tgt_language', self.translation_group.tgt_lang_combo.currentText()))
            except Exception:
                pass
            try:
                legacy_summary = settings.get('summary_model', self.summary_group.summary_combo.currentText())
                if legacy_summary in ["MT5 (lokalny)", "PLT5 (lokalny)"]:
                    legacy_summary = "BART (lokalny)"
                self.summary_group.summary_combo.setCurrentText(legacy_summary)
            except Exception:
                pass
            try:
                self.summary_group.summary_lang_combo.setCurrentText(settings.get('summary_language', self.summary_group.summary_lang_combo.currentText()))
            except Exception:
                pass
            # Apply formats checkboxes
            for cb in getattr(self.formats_group, 'original_checkboxes', []):
                cb.setChecked(settings.get('formats_original', []).count(cb.text()) > 0)
            for cb in getattr(self.formats_group, 'translated_checkboxes', []):
                cb.setChecked(settings.get('formats_translated', []).count(cb.text()) > 0)
            for cb in getattr(self.formats_group, 'summary_checkboxes', []):
                cb.setChecked(settings.get('formats_summary', []).count(cb.text()) > 0)
        except Exception:
            pass
        self._update_ui_state()

    def _gather_settings(self) -> dict:
        """Gather current UI settings into a serializable dict."""
        try:
            return {
                'transcription_model': self.transcription_group.model_combo.currentText(),
                'transcription_src_language': self.transcription_group.src_lang_combo.currentText(),
                'whisper_variant': self.whisper_variant,
                'whisper_device': self.whisper_device,
                'whisper_device_index': self.whisper_device_index,

                'nllb_variant': self.nllb_variant,
                'nllb_device': self.nllb_device,
                'nllb_device_index': self.nllb_device_index,
                'hf_summary_model_name': self.hf_summary_model_name,
                'hf_summary_device': self.hf_summary_device,
                'hf_summary_device_index': self.hf_summary_device_index,
                'hf_summary_max_length': self.hf_summary_max_length,
                'hf_summary_min_length': self.hf_summary_min_length,
                'hf_summary_num_beams': self.hf_summary_num_beams,
                'bart_summary_prompt': self.bart_summary_prompt,
                'ollama_model_name': self.ollama_model_name,
                'ollama_translation_prompt': self.ollama_translation_prompt,
                'translation_openrouter_model_name': self.translation_openrouter_model_name,
                'translation_openrouter_prompt': self.translation_openrouter_prompt,
                'translation_model': self.translation_group.translation_combo.currentText(),
                'translation_src_language': self.translation_group.translation_src_lang_combo.currentText(),
                'translation_tgt_language': self.translation_group.tgt_lang_combo.currentText(),
                'summary_model': self.summary_group.summary_combo.currentText(),
                'summary_language': self.summary_group.summary_lang_combo.currentText(),
                'ollama_summary_model_name': self.ollama_summary_model_name,
                'ollama_summary_prompt': self.ollama_summary_prompt,
                'summary_gemini_prompt': self.summary_gemini_prompt,
                'summary_openrouter_prompt': self.summary_openrouter_prompt,
                'whisper_delete_audio': self.whisper_delete_audio,
                'whisper_enable_paragraphing': self.whisper_enable_paragraphing,
                'enable_denoising': self.whisper_enable_denoising,
                'enable_normalization': self.whisper_enable_normalization,
                'force_mono': self.whisper_force_mono,
                'transcription_segment_batch_size': self.transcription_segment_batch_size,
                'translation_segment_batch_size': self.translation_segment_batch_size,
                'enable_diarization': self.enable_diarization,
                'diarization_num_speakers': self.diarization_num_speakers,

                'formats_original': [cb.text() for cb in getattr(self.formats_group, 'original_checkboxes', []) if cb.isChecked()],
                'formats_translated': [cb.text() for cb in getattr(self.formats_group, 'translated_checkboxes', []) if cb.isChecked()],
                'formats_summary': [cb.text() for cb in getattr(self.formats_group, 'summary_checkboxes', []) if cb.isChecked()],
                
                'transcription_correction': self.transcription_group.correction_combo.currentText(),
                'correction_ollama_model_name': self.correction_ollama_model_name,
                'correction_prompt': self.correction_prompt,
                'gemini_key': self.gemini_key,
                'openrouter_key': self.openrouter_key,
                'openrouter_model_name': self.openrouter_model_name,
                'summary_openrouter_model_name': self.summary_openrouter_model_name,
                'enable_forced_alignment': self.enable_forced_alignment,
                'forced_alignment_model': self.forced_alignment_model
            }
        except Exception:
            return {}

    def _save_current_settings(self):
        # If we're applying settings or otherwise suppressing saves, skip persisting
        if getattr(self, '_suppress_save', False):
            return
        try:
            settings = self._gather_settings()
            if save_settings(settings):
                self.append_log("Zapisano ustawienia użytkownika.", "info")
        except Exception:
            self.append_log("Nie udało się zapisać ustawień użytkownika.", "warning")

    def _log_environment_info(self):
        """Log some useful environment and dependency info to help debugging."""
        import tempfile
        import importlib
        # temp dir
        td = tempfile.gettempdir()
        self.append_log(f"Temp dir: {td}", "info")

        # downloads dir in project if exists
        downloads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'downloads')
        if os.path.exists(downloads_dir):
            try:
                n_files = len([f for f in os.listdir(downloads_dir) if os.path.isfile(os.path.join(downloads_dir, f))])
                self.append_log(f"Downloads folder: {downloads_dir} (files={n_files})", "info")
            except Exception:
                self.append_log(f"Downloads folder present: {downloads_dir}", "info")

        # Key library versions
        libs = ['torch', 'ctranslate2', 'httpx', 'faster_whisper']
        for lib in libs:
            try:
                m = importlib.import_module(lib)
                ver = getattr(m, '__version__', None) or getattr(m, 'VERSION', None) or str(m)
                self.append_log(f"Lib {lib}: {ver}", "info")
            except Exception:
                self.append_log(f"Lib {lib}: not installed", "warning")

        # CUDA devices info (if any were already enumerated)
        try:
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                count = torch.cuda.device_count()
                self.append_log(f"Szczegóły CUDA: znaleziono {count} urządzeń.", "info")
                for i in range(count):
                    try:
                        name = torch.cuda.get_device_name(i)
                        self.append_log(f"  index={i} name={name}", "debug")
                    except Exception:
                        self.append_log(f"  index={i} name=unknown", "warning")
        except Exception:
            pass

    def start_transcription(self):
        url = self.source_group.url_entry.text().strip()
        if not url and not self.local_files:
            QMessageBox.critical(self, "Błąd", "Podaj adres URL lub wybierz plik lokalny.")
            return

        # Simplified config creation
        config = self.get_transcription_config()
        if not config:
            return
        
        if config.enable_diarization and not config.hf_token:
            self.append_log("Błąd: Diaryzacja jest włączona, ale nie podano tokenu Hugging Face.", "error")
            self.open_diarization_settings()
            return
            
        try:
            cfg_summary = (
                f"Transcription config: model={config.transcription_model}, whisper_variant={config.whisper_variant}, "
                f"device={config.whisper_device}, device_index={config.whisper_device_index}, "
                f"translation_model={config.translation_model}, nllb_variant={config.nllb_variant}, "
                f"summary_model={config.summary_model}"
            )
            self.append_log(cfg_summary, "info")
        except Exception:
            self.append_log("Uruchomienie: nie udało się wygenerować podsumowania konfiguracji.", "warning")

        self.progress_bar.setValue(0)
        self.current_eta = "--:--"
        self.progress_bar.setFormat(f"%p% | ETA: {self.current_eta}")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = TranscriptionThread(config)
        self.thread.progress_signal.connect(self._on_progress_updated)
        self.thread.status_signal.connect(self.append_log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    @Slot(int)
    def _on_progress_updated(self, value: int):
        try:
            self.progress_bar.setValue(int(value))
            self.progress_bar.setFormat(f"%p% | ETA: {self.current_eta}")
        except Exception:
            self.progress_bar.setValue(value)


    def get_transcription_config(self):
        # Language selections
        src_lang_code = self.transcription_group.languages.get(self.transcription_group.src_lang_combo.currentText(), "auto")
        translation_src_lang_code = self.translation_group.languages.get(self.translation_group.translation_src_lang_combo.currentText(), "auto")
        tgt_lang_code = self.translation_group.languages.get(self.translation_group.tgt_lang_combo.currentText())
        summary_lang_code = self.summary_group.languages.get(self.summary_group.summary_lang_combo.currentText())

        return TranscriptionConfig(
            url=self.source_group.url_entry.text().strip(),
            local_files=self.local_files,
            transcription_model=self.transcription_group.model_combo.currentText(),
            whisper_variant=self.whisper_variant,
            whisper_device=self.whisper_device,
            whisper_device_index=self.whisper_device_index,
            translation_model=self.translation_group.translation_combo.currentText(),
            translation_openrouter_model_name=self.translation_openrouter_model_name,
            translation_openrouter_prompt=self.translation_openrouter_prompt,
            translation_ollama_prompt=self.ollama_translation_prompt,
            translation_segment_batch_size=self.translation_segment_batch_size,
            nllb_variant=self.nllb_variant,
            nllb_device=self.nllb_device,
            nllb_device_index=self.nllb_device_index,
            hf_summary_model_name=self.hf_summary_model_name,
            hf_summary_device=self.hf_summary_device,
            hf_summary_device_index=self.hf_summary_device_index,
            hf_summary_max_length=self.hf_summary_max_length,
            hf_summary_min_length=self.hf_summary_min_length,
            hf_summary_num_beams=self.hf_summary_num_beams,
            ollama_model_name=self.ollama_model_name,
            # CTranslate2 summarization options (defaults)
            ctranslate2_device_index=0,
            ctranslate2_tokenizer_name="mtj/bart-base-polish-summarization",
            ctranslate2_max_input_tokens=1024,
            ctranslate2_max_decoding_length=256,
            ctranslate2_beam_size=4,
            summary_model=self.summary_group.summary_combo.currentText(),
            ollama_summary_model_name=self.ollama_summary_model_name,
            ollama_summary_prompt=self.ollama_summary_prompt,
            summary_gemini_prompt=self.summary_gemini_prompt,
            summary_openrouter_prompt=self.summary_openrouter_prompt,
            summary_openrouter_model_name=self.summary_openrouter_model_name,
            bart_summary_prompt=self.bart_summary_prompt,
            # Correction / post-editing options
            transcription_correction=self.transcription_group.correction_combo.currentText(),
            correction_ollama_model_name=self.correction_ollama_model_name,
            correction_prompt=self.correction_prompt,
            transcription_segment_batch_size=self.transcription_segment_batch_size,
            src_lang_code=src_lang_code,
            translation_src_lang_code=translation_src_lang_code,
            tgt_lang_code=tgt_lang_code,
            summary_lang_code=summary_lang_code,
            formats_original=[cb.text() for cb in self.formats_group.original_checkboxes if cb.isChecked()],
            formats_translated=[cb.text() for cb in self.formats_group.translated_checkboxes if cb.isChecked()],
            formats_summary=[cb.text() for cb in self.formats_group.summary_checkboxes if cb.isChecked()],
            openai_key=None,
            gemini_key=self.gemini_key,
            openrouter_key=self.openrouter_key,
            openrouter_model_name=self.openrouter_model_name,
            delete_audio=self.whisper_delete_audio,
            enable_paragraphing=self.whisper_enable_paragraphing,
            # New diarization
            enable_diarization=self.enable_diarization,
            hf_token=self.diarization_hf_token if self.enable_diarization else None,
            num_speakers=self.diarization_num_speakers if self.enable_diarization else 0,
            enable_denoising=self.whisper_enable_denoising,
            enable_normalization=self.whisper_enable_normalization,
            force_mono=self.whisper_force_mono
        )

    def on_finished(self, msg, msg_type):
        QMessageBox.information(self, "Gotowe", msg) if msg_type == "success" else QMessageBox.critical(self, "Błąd", msg)
        self.current_eta = "--:--"
        self.progress_bar.setFormat("%p%")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)