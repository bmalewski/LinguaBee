import httpx
import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, QSpinBox, QVBoxLayout, QGridLayout, QMessageBox, QDialog, QDialogButtonBox, QHBoxLayout, QGroupBox, QFormLayout, QTextEdit, QInputDialog)

# ApiKeyDialog removed: OpenAI/Gemini API key dialogs are no longer used in the GUI per user request.

class WhisperSettingsDialog(QDialog):
    def __init__(self, parent, available_devices, current_variant, current_device, current_device_index, current_delete_audio, current_enable_paragraphing, current_enable_denoising, current_enable_normalization, current_force_mono, current_enable_diarization, current_num_speakers, current_transcription_segment_batch_size=250):
        super().__init__(parent) # Poprawka: przekazanie argumentów do konstruktora
        self.setWindowTitle("Ustawienia Whisper")
        self.available_devices = available_devices

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # --- Transcription Settings ---
        transcription_group = QGroupBox("Transkrypcja")
        transcription_layout = QFormLayout()

        self.variant_combo = QComboBox()
        self.variant_combo.addItems(["medium", "large-v3", "turbo"])
        self.variant_combo.setCurrentText(current_variant)
        transcription_layout.addRow("Wariant modelu:", self.variant_combo)


        self.device_combo = QComboBox()
        for display_name, data in self.available_devices:
            # store device metadata in the combo item's userData so we can match and restore selection
            self.device_combo.addItem(display_name, userData=data)
        
        for i in range(self.device_combo.count()):
            data = self.device_combo.itemData(i)
            if data and data['device'] == current_device and data.get('device_index', 0) == current_device_index:
                self.device_combo.setCurrentIndex(i)
                break
        if self.device_combo.currentIndex() == -1:
            self.device_combo.setCurrentIndex(0)
        transcription_layout.addRow("Urządzenie:", self.device_combo)
        transcription_group.setLayout(transcription_layout)
        form_layout.addWidget(transcription_group)

        # --- Audio Processing Settings ---
        audio_group = QGroupBox("Przetwarzanie Audio")
        audio_layout = QVBoxLayout()
        self.denoising_checkbox = QCheckBox("Włącz odszumianie (może poprawić jakość na zaszumionych nagraniach)")
        self.denoising_checkbox.setChecked(current_enable_denoising)
        audio_layout.addWidget(self.denoising_checkbox)

        self.normalization_checkbox = QCheckBox("Normalizuj głośność audio (wyrównuje poziomy dźwięku)")
        self.normalization_checkbox.setChecked(current_enable_normalization)
        audio_layout.addWidget(self.normalization_checkbox)

        self.mono_checkbox = QCheckBox("Konwertuj do mono (zalecane dla diaryzacji)")
        self.mono_checkbox.setChecked(current_force_mono)
        audio_layout.addWidget(self.mono_checkbox)
        audio_group.setLayout(audio_layout)
        form_layout.addWidget(audio_group)

        # --- Diarization Settings ---
        diarization_group = QGroupBox("Diaryzacja (Rozpoznawanie mówców)")
        diarization_group.setCheckable(True)
        diarization_group.setChecked(current_enable_diarization)
        diarization_layout = QFormLayout()
        
        self.num_speakers_label = QLabel("Liczba mówców (0 = auto):")
        self.num_speakers_spinbox = QSpinBox()
        self.num_speakers_spinbox.setRange(0, 20)
        self.num_speakers_spinbox.setValue(current_num_speakers)
        diarization_layout.addRow(self.num_speakers_label, self.num_speakers_spinbox)
        
        diarization_group.setLayout(diarization_layout)
        form_layout.addWidget(diarization_group)

        # Connect signals for diarization group
        diarization_group.toggled.connect(self.num_speakers_label.setEnabled)
        diarization_group.toggled.connect(self.num_speakers_spinbox.setEnabled)
        self.num_speakers_label.setEnabled(current_enable_diarization)
        self.num_speakers_spinbox.setEnabled(current_enable_diarization)
        
        # --- Post-processing Settings ---
        post_proc_group = QGroupBox("Opcje dodatkowe")
        post_proc_layout = QVBoxLayout()
        self.delete_audio_checkbox = QCheckBox("Usuń plik audio po zakończeniu (dot. pobrań z URL)")
        self.delete_audio_checkbox.setChecked(current_delete_audio)
        post_proc_layout.addWidget(self.delete_audio_checkbox)

        self.paragraphing_checkbox = QCheckBox("Podziel na akapity (jeśli diaryzacja jest wyłączona)")
        self.paragraphing_checkbox.setChecked(current_enable_paragraphing)
        post_proc_layout.addWidget(self.paragraphing_checkbox)

        self.transcription_segment_batch_spin = QSpinBox()
        self.transcription_segment_batch_spin.setRange(1, 1000)
        self.transcription_segment_batch_spin.setValue(int(current_transcription_segment_batch_size or 250))
        post_proc_layout.addWidget(QLabel("Paczka segmentów SRT (korekta):"))
        post_proc_layout.addWidget(self.transcription_segment_batch_spin)
        
        post_proc_group.setLayout(post_proc_layout)
        form_layout.addWidget(post_proc_group)

        self.diarization_group = diarization_group # Store reference

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self):
        # Find the data for the selected device
        # Prefer the combo's stored userData; fall back to available_devices list
        selected_device_data = self.device_combo.currentData()
        if not selected_device_data:
            idx = self.device_combo.currentIndex()
            if 0 <= idx < len(self.available_devices):
                selected_device_data = self.available_devices[idx][1]

        return (
            self.variant_combo.currentText(),
            selected_device_data,
            self.delete_audio_checkbox.isChecked(),
            self.paragraphing_checkbox.isChecked(),
            self.denoising_checkbox.isChecked(),
            self.normalization_checkbox.isChecked(),
            self.mono_checkbox.isChecked(),
            self.diarization_group.isChecked(),
            self.num_speakers_spinbox.value(),
            self.transcription_segment_batch_spin.value()
        )

class NllbSettingsDialog(QDialog):
    def __init__(self, parent=None, available_devices=None, current_variant="1.3B", current_device="cpu", current_device_index=0, current_translation_segment_batch_size=250):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia NLLB")
        
        self.layout = QGridLayout(self)
        
        self.layout.addWidget(QLabel("Typ modelu:"), 0, 0)
        self.variant_combo = QComboBox()
        # The application supports a few NLLB variants. The 12B variant was removed
        # from the UI to avoid users accidentally attempting large-model downloads.
        self.variant_combo.addItems(["distilled-600M", "1.3B", "3.3B"])
        self.variant_combo.setCurrentText(current_variant)
        self.layout.addWidget(self.variant_combo, 0, 1)
        
        self.layout.addWidget(QLabel("Urządzenie:"), 1, 0)
        self.device_combo = QComboBox()
        if available_devices:
            for name, data in available_devices:
                self.device_combo.addItem(name, userData=data)

        # Find and set the current device
        for i in range(self.device_combo.count()):
            data = self.device_combo.itemData(i)
            if data['device'] == current_device and data.get('device_index', 0) == current_device_index:
                self.device_combo.setCurrentIndex(i)
                break
        self.layout.addWidget(self.device_combo, 1, 1)

        self.layout.addWidget(QLabel("Paczka segmentów SRT (tłumaczenie):"), 2, 0)
        self.translation_segment_batch_spin = QSpinBox()
        self.translation_segment_batch_spin.setRange(1, 1000)
        self.translation_segment_batch_spin.setValue(int(current_translation_segment_batch_size or 250))
        self.layout.addWidget(self.translation_segment_batch_spin, 2, 1)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 3, 0, 1, 2)

    def get_settings(self):
        device_settings = self.device_combo.currentData()
        return self.variant_combo.currentText(), device_settings, self.translation_segment_batch_spin.value()

class BartSummarizationSettingsDialog(QDialog):
    def __init__(self, parent=None, available_devices=None, current_model="mtj/bart-base-polish-summarization", current_device="cpu", current_device_index=0, current_max_length=150, current_min_length=30, current_num_beams=4, current_prompt=""):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Streszczenia (BART)")

        self.layout = QGridLayout(self)

        self.layout.addWidget(QLabel("Model BART:"), 0, 0)
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(["mtj/bart-base-polish-summarization"])
        self.variant_combo.setCurrentText(current_model or "mtj/bart-base-polish-summarization")
        self.layout.addWidget(self.variant_combo, 0, 1, 1, 3)

        self.layout.addWidget(QLabel("Urządzenie:"), 1, 0)
        self.device_combo = QComboBox()
        if available_devices:
            for name, data in available_devices:
                self.device_combo.addItem(name, userData=data)
        for i in range(self.device_combo.count()):
            data = self.device_combo.itemData(i)
            if data and data['device'] == current_device and data.get('device_index', 0) == current_device_index:
                self.device_combo.setCurrentIndex(i)
                break
        self.layout.addWidget(self.device_combo, 1, 1, 1, 3)

        self.layout.addWidget(QLabel("Max długość:"), 2, 0)
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(20, 2048)
        self.max_length_spin.setValue(current_max_length)
        self.layout.addWidget(self.max_length_spin, 2, 1)

        self.layout.addWidget(QLabel("Min długość:"), 2, 2)
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(0, 1024)
        self.min_length_spin.setValue(current_min_length)
        self.layout.addWidget(self.min_length_spin, 2, 3)

        self.layout.addWidget(QLabel("Liczba beamów:"), 3, 0)
        self.num_beams_spin = QSpinBox()
        self.num_beams_spin.setRange(1, 16)
        self.num_beams_spin.setValue(current_num_beams)
        self.layout.addWidget(self.num_beams_spin, 3, 1)

        self.layout.addWidget(QLabel("Szablony promptów:"), 4, 0)
        self.template_combo = QComboBox()
        self.layout.addWidget(self.template_combo, 4, 1)
        self.save_template_btn = QPushButton("Zapisz szablon")
        self.save_template_btn.clicked.connect(self._save_template)
        self.layout.addWidget(self.save_template_btn, 4, 2)
        self.delete_template_btn = QPushButton("Usuń szablon")
        self.delete_template_btn.clicked.connect(self._delete_template)
        self.layout.addWidget(self.delete_template_btn, 4, 3)

        self.layout.addWidget(QLabel("Prompt (streszczenie BART):"), 5, 0)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Wprowadź prompt do streszczenia. Użyj {text} jako placeholderu na treść.")
        self.prompt_edit.setMinimumHeight(220)
        self.prompt_edit.setPlainText(current_prompt or "")
        self.layout.addWidget(self.prompt_edit, 6, 0, 1, 4)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 7, 0, 1, 4)

        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'summary_bart')
        os.makedirs(self.prompts_dir, exist_ok=True)
        self.template_combo.currentTextChanged.connect(self._on_template_selected)
        self._load_templates()

    def _list_prompt_files(self):
        try:
            files = [f for f in os.listdir(self.prompts_dir) if os.path.isfile(os.path.join(self.prompts_dir, f)) and f.lower().endswith('.txt')]
            files.sort()
            return files
        except Exception:
            return []

    def _load_templates(self):
        current_prompt = self.prompt_edit.toPlainText().strip()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        files = self._list_prompt_files()
        for fn in files:
            self.template_combo.addItem(os.path.splitext(fn)[0])
        self.template_combo.blockSignals(False)

        # Select matching template name (without loading/overwriting prompt).
        if current_prompt:
            for fn in files:
                name = os.path.splitext(fn)[0]
                fp = os.path.join(self.prompts_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        if fh.read().strip() == current_prompt:
                            self.template_combo.blockSignals(True)
                            self.template_combo.setCurrentText(name)
                            self.template_combo.blockSignals(False)
                            break
                except Exception:
                    continue

    def _on_template_selected(self, name: str):
        if not name:
            return
        filename = os.path.join(self.prompts_dir, f"{name}.txt")
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as fh:
                    self.prompt_edit.setPlainText(fh.read())
        except Exception:
            pass

    def _sanitize_filename(self, name: str) -> str:
        import re
        s = name.strip().replace(' ', '_')
        s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
        return s or 'prompt_bart'

    def _save_template(self):
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text or len(prompt_text) < 20:
            QMessageBox.warning(self, "Za krótki prompt", "Prompt musi mieć co najmniej 20 znaków.")
            return
        name, ok = QInputDialog.getText(self, "Nazwa szablonu", "Podaj nazwę szablonu:")
        if not ok or not name.strip():
            return
        safe = self._sanitize_filename(name)
        try:
            with open(os.path.join(self.prompts_dir, f"{safe}.txt"), 'w', encoding='utf-8') as fh:
                fh.write(prompt_text)
            self._load_templates()
            self.template_combo.setCurrentText(safe)
            QMessageBox.information(self, "Zapisano", f"Zapisano szablon '{safe}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się zapisać szablonu: {e}")

    def _delete_template(self):
        name = self.template_combo.currentText()
        if not name:
            return
        ok = QMessageBox.question(self, "Usuń szablon", f"Czy na pewno chcesz usunąć szablon '{name}'?")
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            fp = os.path.join(self.prompts_dir, f"{name}.txt")
            if os.path.exists(fp):
                os.remove(fp)
            self._load_templates()
            QMessageBox.information(self, "Usunięto", f"Usunięto szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się usunąć szablonu: {e}")

    def get_settings(self):
        device_settings = self.device_combo.currentData() or {'device': 'cpu', 'device_index': 0}
        return {
            "model_name": self.variant_combo.currentText(),
            "device": device_settings['device'],
            "device_index": device_settings.get('device_index', 0),
            "max_length": self.max_length_spin.value(),
            "min_length": self.min_length_spin.value(),
            "num_beams": self.num_beams_spin.value(),
            "prompt": self.prompt_edit.toPlainText().strip(),
        }

class OllamaSettingsDialog(QDialog):
    def __init__(self, parent=None, current_model=""):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Ollama")
        
        self.layout = QGridLayout(self)
        
        self.layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        # Use a non-editable dropdown so users pick from available models (or prepopulated suggestions)
        self.model_combo.setEditable(False)
        common_models = [
            "gpt-4o-mini",
            "gpt-oss:7b",
            "gpt-oss:13b",
            "vicuna:13b",
            "gpt-oss:120b-cloud",
            "mistral:7b",
        ]
        self.model_combo.addItems(common_models)
        self.layout.addWidget(self.model_combo, 0, 1)

        self.layout.addWidget(QLabel("Urządzenie:"), 1, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto"])
        self.device_combo.setEnabled(False)
        self.device_combo.setToolTip("Urządzenie jest konfigurowane globalnie w aplikacji Ollama.")
        self.layout.addWidget(self.device_combo, 1, 1)
        
        self.refresh_btn = QPushButton("Odśwież listę modeli")
        self.refresh_btn.clicked.connect(self.populate_models)
        self.layout.addWidget(self.refresh_btn, 2, 0, 1, 2)

        # Standard OK / Cancel for Ollama settings
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 3, 0, 1, 2)

        self.populate_models(current_model)

class CorrectionSettingsDialog(QDialog):
    """Dialog do konfiguracji opcji Korekta (Ollama + prompt).

    Templates are persisted as individual text files under a `prompts/` directory
    located in the project root. Selecting a template loads its contents into the
    prompt editor. Saving/deleting templates operates on files in that folder.
    """
    def __init__(self, parent=None, current_model="", current_prompt=""):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Korekty (Ollama)")

        self.layout = QGridLayout(self)

        # Row 0: Model label, model combo, refresh button (refresh to the right of combo)
        self.layout.addWidget(QLabel("Model Ollama:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        # prepopulate with common suggestions
        suggestions = [
            "gpt-4o-mini",
            "gpt-oss:7b",
            "gpt-oss:13b",
            "vicuna:13b",
            "mistral:7b",
        ]
        self.model_combo.addItems(suggestions)
        self.layout.addWidget(self.model_combo, 0, 1)

        self.refresh_btn = QPushButton("Odśwież listę modeli")
        self.refresh_btn.clicked.connect(self.populate_models)
        # place refresh to the right of the model combo
        self.layout.addWidget(self.refresh_btn, 0, 2)

        # Row 1: Szablony promptów label below model label, and template combo + buttons shifted left
        self.layout.addWidget(QLabel("Szablony promptów:"), 1, 0)
        self.template_combo = QComboBox()
        self.layout.addWidget(self.template_combo, 1, 1)

        self.save_template_btn = QPushButton("Zapisz szablon")
        self.save_template_btn.clicked.connect(self._save_template)
        self.layout.addWidget(self.save_template_btn, 1, 2)

        self.delete_template_btn = QPushButton("Usuń szablon")
        self.delete_template_btn.clicked.connect(self._delete_template)
        self.layout.addWidget(self.delete_template_btn, 1, 3)

        # Row 2: Prompt label under Szablony
        self.layout.addWidget(QLabel("Prompt (korekta):"), 2, 0)
        # Row 3: Large prompt field under the label, spanning columns to preserve width
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Wprowadź prompt do korekty...")
        # increase prompt height by ~50% (was 160)
        self.prompt_edit.setMinimumHeight(240)
        self.layout.addWidget(self.prompt_edit, 3, 0, 1, 4)
        # Buttons placed below the prompt (row 4). Customize labels and colors for Correction dialog.
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.Ok)
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        if ok_btn:
            ok_btn.setText("OK")
            ok_btn.setStyleSheet("background-color: #1976D2; color: white; padding:6px 12px;")
        if cancel_btn:
            cancel_btn.setText("ANULUJ")
            cancel_btn.setStyleSheet("background-color: #D32F2F; color: white; padding:6px 12px;")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # place buttons in the row below the prompt, spanning full width
        self.layout.addWidget(buttons, 4, 0, 1, 4)

        # templates directory (project_root/prompts)
        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
        os.makedirs(self.prompts_dir, exist_ok=True)

        # wire up template selection to load file contents
        self.template_combo.currentTextChanged.connect(self._on_template_selected)

        # populate with current values and templates
        self._load_templates()
        self.populate_models(current_model)
        if current_prompt:
            # if provided current_prompt looks like a filename in prompts/, load that file
            try:
                # prefer exact filename match
                candidate = current_prompt
                if os.path.isabs(candidate) and os.path.exists(candidate):
                    with open(candidate, 'r', encoding='utf-8') as fh:
                        self.prompt_edit.setPlainText(fh.read())
                else:
                    # otherwise, just set the raw text
                    self.prompt_edit.setPlainText(current_prompt)
            except Exception:
                self.prompt_edit.setPlainText(current_prompt)

    def populate_models(self, current_model=""):
        existing = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        self.model_combo.clear()
        self.model_combo.setEnabled(False)
        self.model_combo.addItem("Pobieranie modeli...")
        QApplication.processEvents()
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models if m.get("name")]
            merged = []
            for m in model_names:
                if m not in merged:
                    merged.append(m)
            for s in existing:
                if s and s not in merged:
                    merged.append(s)
            self.model_combo.clear()
            if merged:
                self.model_combo.addItems(merged)
                if current_model and current_model in merged:
                    self.model_combo.setCurrentText(current_model)
            else:
                self.model_combo.addItem("Brak modeli Ollama")
        except (httpx.ConnectError, httpx.ReadTimeout):
            # Keep suggestions if Ollama not available
            self.model_combo.clear()
            if existing:
                for s in existing:
                    self.model_combo.addItem(s)
            else:
                for m in ["gpt-4o-mini", "gpt-oss:7b", "gpt-oss:13b", "vicuna:13b", "mistral:7b"]:
                    self.model_combo.addItem(m)
            if current_model:
                self.model_combo.setCurrentText(current_model)
        except Exception as e:
            self.model_combo.clear()
            self.model_combo.addItem(f"Błąd: {e}")
        finally:
            self.model_combo.setEnabled(True)

    def get_settings(self):
        model = self.model_combo.currentText()
        if model.startswith("Błąd:") or model == "Brak modeli Ollama":
            model = ""
        prompt = self.prompt_edit.toPlainText().strip()
        return model, prompt

    def _list_prompt_files(self):
        try:
            files = [f for f in os.listdir(self.prompts_dir) if os.path.isfile(os.path.join(self.prompts_dir, f)) and f.lower().endswith('.txt')]
            files.sort()
            return files
        except Exception:
            return []

    def _load_templates(self):
        current_prompt = self.prompt_edit.toPlainText().strip()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        files = self._list_prompt_files()
        self.template_names = []
        for fn in files:
            name = os.path.splitext(fn)[0]
            self.template_combo.addItem(name)
            self.template_names.append(name)
        self.template_combo.blockSignals(False)

        # Select matching template name (without loading/overwriting prompt).
        if current_prompt:
            for fn in files:
                name = os.path.splitext(fn)[0]
                fp = os.path.join(self.prompts_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        if fh.read().strip() == current_prompt:
                            self.template_combo.blockSignals(True)
                            self.template_combo.setCurrentText(name)
                            self.template_combo.blockSignals(False)
                            break
                except Exception:
                    continue

    def _on_template_selected(self, name: str):
        if not name:
            return
        filename = os.path.join(self.prompts_dir, f"{name}.txt")
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as fh:
                    self.prompt_edit.setPlainText(fh.read())
        except Exception:
            pass

    def _sanitize_filename(self, name: str) -> str:
        import re
        s = name.strip().replace(' ', '_')
        s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
        if not s:
            s = 'prompt'
        return s

    def _save_template(self):
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text or len(prompt_text) < 20:
            QMessageBox.warning(self, "Za krótki prompt", "Prompt musi mieć co najmniej 20 znaków.")
            return
        name, ok = QInputDialog.getText(self, "Nazwa szablonu", "Podaj nazwę szablonu:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            safe = self._sanitize_filename(name)
            filename = os.path.join(self.prompts_dir, f"{safe}.txt")
            with open(filename, 'w', encoding='utf-8') as fh:
                fh.write(prompt_text)
            self._load_templates()
            try:
                self.template_combo.setCurrentText(name)
            except Exception:
                self.template_combo.setCurrentText(safe)
            QMessageBox.information(self, "Zapisano", f"Zapisano szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się zapisać szablonu: {e}")

    def _delete_template(self):
        name = self.template_combo.currentText()
        if not name:
            return
        ok = QMessageBox.question(self, "Usuń szablon", f"Czy na pewno chcesz usunąć szablon '{name}'?")
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            filename = os.path.join(self.prompts_dir, f"{name}.txt")
            if os.path.exists(filename):
                os.remove(filename)
            self._load_templates()
            QMessageBox.information(self, "Usunięto", f"Usunięto szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się usunąć szablonu: {e}")


class OllamaSummarySettingsDialog(CorrectionSettingsDialog):
    """Dialog ustawień streszczeń Ollama: model + prompt + szablony (bez pól Gemini)."""
    def __init__(self, parent=None, current_model="", current_prompt=""):
        super().__init__(parent=parent, current_model=current_model, current_prompt=current_prompt)
        self.setWindowTitle("Ustawienia Streszczenia (Ollama)")
        self.prompt_edit.setPlaceholderText("Wprowadź prompt do streszczenia Ollama... Użyj {text}, {language_name}, {language_code}.")
        original_prompt = current_prompt or self.prompt_edit.toPlainText()

        # wspólny katalog szablonów dla sekcji streszczenie (Ollama + Gemini)
        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'summary')
        os.makedirs(self.prompts_dir, exist_ok=True)
        self._load_templates()
        # Nie nadpisuj promptu użytkownika po przeładowaniu listy szablonów.
        if isinstance(original_prompt, str) and original_prompt.strip():
            self.prompt_edit.setPlainText(original_prompt)


class OllamaTranslationSettingsDialog(CorrectionSettingsDialog):
    """Dialog ustawień tłumaczenia Ollama: model + prompt + szablony."""
    def __init__(self, parent=None, current_model="", current_prompt="", current_translation_segment_batch_size=250):
        super().__init__(parent=parent, current_model=current_model, current_prompt=current_prompt)
        self.setWindowTitle("Ustawienia Tłumaczenia (Ollama)")
        self.prompt_edit.setPlaceholderText(
            "Wprowadź prompt do tłumaczenia Ollama... Użyj {text}, {src_lang}, {tgt_lang}."
        )
        original_prompt = current_prompt or self.prompt_edit.toPlainText()

        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'translation')
        os.makedirs(self.prompts_dir, exist_ok=True)
        self._load_templates()
        self.layout.addWidget(QLabel("Paczka segmentów SRT (tłumaczenie):"), 5, 0)
        self.translation_segment_batch_spin = QSpinBox()
        self.translation_segment_batch_spin.setRange(1, 1000)
        self.translation_segment_batch_spin.setValue(int(current_translation_segment_batch_size or 250))
        self.layout.addWidget(self.translation_segment_batch_spin, 5, 1, 1, 3)
        if isinstance(original_prompt, str) and original_prompt.strip():
            self.prompt_edit.setPlainText(original_prompt)

    def get_settings(self):
        model, prompt = super().get_settings()
        return model, prompt, self.translation_segment_batch_spin.value()

class GeminiCorrectionSettingsDialog(QDialog):
    """Ustawienia korekty dla Gemini: tylko klucz API + prompt (bez modeli Ollama)."""
    def __init__(self, parent=None, current_key="", current_prompt="", current_transcription_segment_batch_size=250):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Korekty (Gemini API)")
        self.layout = QGridLayout(self)

        self.layout.addWidget(QLabel("Klucz API Gemini:"), 0, 0)
        self.key_input = QLineEdit(current_key)
        self.key_input.setPlaceholderText("Wklej klucz API Gemini...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.key_input, 0, 1, 1, 3)

        self.layout.addWidget(QLabel("Prompt (korekta):"), 1, 0)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Wprowadź prompt do korekty dla Gemini...")
        self.prompt_edit.setMinimumHeight(240)
        self.prompt_edit.setPlainText(current_prompt or "")
        self.layout.addWidget(self.prompt_edit, 2, 0, 1, 4)

        self.layout.addWidget(QLabel("Paczka segmentów SRT (korekta):"), 3, 0)
        self.transcription_segment_batch_spin = QSpinBox()
        self.transcription_segment_batch_spin.setRange(1, 1000)
        self.transcription_segment_batch_spin.setValue(int(current_transcription_segment_batch_size or 250))
        self.layout.addWidget(self.transcription_segment_batch_spin, 3, 1, 1, 3)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 4, 0, 1, 4)

    def get_settings(self):
        return self.key_input.text().strip(), self.prompt_edit.toPlainText().strip(), self.transcription_segment_batch_spin.value()


class OpenRouterCorrectionSettingsDialog(QDialog):
    """Ustawienia korekty dla OpenRouter: klucz API + prompt."""
    def __init__(self, parent=None, current_key="", current_prompt="", current_model="google/gemini-2.5-flash", current_transcription_segment_batch_size=250):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Korekty (OpenRouter API)")
        self.layout = QGridLayout(self)

        self.layout.addWidget(QLabel("Klucz API OpenRouter:"), 0, 0)
        self.key_input = QLineEdit(current_key)
        self.key_input.setPlaceholderText("Wklej klucz API OpenRouter...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.key_input, 0, 1, 1, 3)

        self.layout.addWidget(QLabel("Model OpenRouter:"), 1, 0)
        self.model_input = QComboBox()
        self.model_input.setEditable(False)
        self.model_input.addItem("Gemini 3 flash (preview)", userData="google/gemini-3-flash-preview")
        self.model_input.addItem("Gemini 2.5 flash", userData="google/gemini-2.5-flash")
        self.model_input.addItem("Gemini 3 pro", userData="google/gemini-3-pro-preview")
        self.model_input.addItem("Gemini 2.5 pro", userData="google/gemini-2.5-pro")
        self.model_input.addItem("GPT-5", userData="openai/gpt-5")
        self.model_input.addItem("GPT-5 mini", userData="openai/gpt-5-mini")
        model_to_set = current_model or "google/gemini-2.5-flash"
        idx = self.model_input.findData(model_to_set)
        if idx >= 0:
            self.model_input.setCurrentIndex(idx)
        else:
            self.model_input.setCurrentIndex(1)
        self.layout.addWidget(self.model_input, 1, 1, 1, 3)

        self.layout.addWidget(QLabel("Paczka segmentów SRT (korekta):"), 2, 0)
        self.transcription_segment_batch_spin = QSpinBox()
        self.transcription_segment_batch_spin.setRange(1, 1000)
        self.transcription_segment_batch_spin.setValue(int(current_transcription_segment_batch_size or 250))
        self.layout.addWidget(self.transcription_segment_batch_spin, 2, 1, 1, 3)

        self.layout.addWidget(QLabel("Prompt (korekta):"), 3, 0)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Wprowadź prompt do korekty dla OpenRouter...")
        self.prompt_edit.setMinimumHeight(240)
        self.prompt_edit.setPlainText(current_prompt or "")
        self.layout.addWidget(self.prompt_edit, 4, 0, 1, 4)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 5, 0, 1, 4)

    def get_settings(self):
        model_id = self.model_input.currentData() or "google/gemini-2.5-flash"
        return self.key_input.text().strip(), self.prompt_edit.toPlainText().strip(), str(model_id).strip(), self.transcription_segment_batch_spin.value()


class OpenRouterSummarySettingsDialog(QDialog):
    """Ustawienia streszczenia dla OpenRouter: klucz API + model + prompt + szablony (wspólne z Ollama/Gemini)."""
    def __init__(self, parent=None, current_key="", current_prompt="", current_model="google/gemini-2.5-flash"):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Streszczenia (OpenRouter API)")
        self.layout = QGridLayout(self)
        original_prompt = current_prompt or ""

        self.layout.addWidget(QLabel("Klucz API OpenRouter:"), 0, 0)
        self.key_input = QLineEdit(current_key)
        self.key_input.setPlaceholderText("Wklej klucz API OpenRouter...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.key_input, 0, 1, 1, 3)

        self.layout.addWidget(QLabel("Model OpenRouter:"), 1, 0)
        self.model_input = QComboBox()
        self.model_input.setEditable(False)
        self.model_input.addItem("Gemini 3 flash (preview)", userData="google/gemini-3-flash-preview")
        self.model_input.addItem("Gemini 2.5 flash", userData="google/gemini-2.5-flash")
        self.model_input.addItem("Gemini 3 pro", userData="google/gemini-3-pro-preview")
        self.model_input.addItem("Gemini 2.5 pro", userData="google/gemini-2.5-pro")
        self.model_input.addItem("GPT-5", userData="openai/gpt-5")
        self.model_input.addItem("GPT-5 mini", userData="openai/gpt-5-mini")
        model_to_set = current_model or "google/gemini-2.5-flash"
        idx = self.model_input.findData(model_to_set)
        if idx >= 0:
            self.model_input.setCurrentIndex(idx)
        else:
            self.model_input.setCurrentIndex(1)
        self.layout.addWidget(self.model_input, 1, 1, 1, 3)

        self.layout.addWidget(QLabel("Szablony promptów:"), 2, 0)
        self.template_combo = QComboBox()
        self.layout.addWidget(self.template_combo, 2, 1)

        self.save_template_btn = QPushButton("Zapisz szablon")
        self.save_template_btn.clicked.connect(self._save_template)
        self.layout.addWidget(self.save_template_btn, 2, 2)

        self.delete_template_btn = QPushButton("Usuń szablon")
        self.delete_template_btn.clicked.connect(self._delete_template)
        self.layout.addWidget(self.delete_template_btn, 2, 3)

        self.layout.addWidget(QLabel("Prompt (streszczenie):"), 3, 0)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Wprowadź prompt do streszczenia dla OpenRouter... Użyj {text}, {language_name}, {language_code}.")
        self.prompt_edit.setMinimumHeight(240)
        self.prompt_edit.setPlainText(current_prompt or "")
        self.layout.addWidget(self.prompt_edit, 4, 0, 1, 4)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 5, 0, 1, 4)

        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'summary')
        os.makedirs(self.prompts_dir, exist_ok=True)
        self.template_combo.currentTextChanged.connect(self._on_template_selected)
        self._load_templates()
        if isinstance(original_prompt, str) and original_prompt.strip():
            self.prompt_edit.setPlainText(original_prompt)

    def get_settings(self):
        model_id = self.model_input.currentData() or "google/gemini-2.5-flash"
        return self.key_input.text().strip(), self.prompt_edit.toPlainText().strip(), str(model_id).strip()

    def _list_prompt_files(self):
        try:
            files = [f for f in os.listdir(self.prompts_dir) if os.path.isfile(os.path.join(self.prompts_dir, f)) and f.lower().endswith('.txt')]
            files.sort()
            return files
        except Exception:
            return []

    def _load_templates(self):
        current_prompt = self.prompt_edit.toPlainText().strip()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        files = self._list_prompt_files()
        for fn in files:
            self.template_combo.addItem(os.path.splitext(fn)[0])
        self.template_combo.blockSignals(False)

        if current_prompt:
            for fn in files:
                name = os.path.splitext(fn)[0]
                fp = os.path.join(self.prompts_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        if fh.read().strip() == current_prompt:
                            self.template_combo.blockSignals(True)
                            self.template_combo.setCurrentText(name)
                            self.template_combo.blockSignals(False)
                            break
                except Exception:
                    continue

    def _on_template_selected(self, name: str):
        if not name:
            return
        filename = os.path.join(self.prompts_dir, f"{name}.txt")
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as fh:
                    self.prompt_edit.setPlainText(fh.read())
        except Exception:
            pass

    def _sanitize_filename(self, name: str) -> str:
        import re
        s = name.strip().replace(' ', '_')
        s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
        if not s:
            s = 'prompt'
        return s

    def _save_template(self):
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text or len(prompt_text) < 20:
            QMessageBox.warning(self, "Za krótki prompt", "Prompt musi mieć co najmniej 20 znaków.")
            return
        name, ok = QInputDialog.getText(self, "Nazwa szablonu", "Podaj nazwę szablonu:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            safe = self._sanitize_filename(name)
            filename = os.path.join(self.prompts_dir, f"{safe}.txt")
            with open(filename, 'w', encoding='utf-8') as fh:
                fh.write(prompt_text)
            self._load_templates()
            try:
                self.template_combo.setCurrentText(name)
            except Exception:
                self.template_combo.setCurrentText(safe)
            QMessageBox.information(self, "Zapisano", f"Zapisano szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się zapisać szablonu: {e}")

    def _delete_template(self):
        name = self.template_combo.currentText()
        if not name:
            return
        ok = QMessageBox.question(self, "Usuń szablon", f"Czy na pewno chcesz usunąć szablon '{name}'?")
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            filename = os.path.join(self.prompts_dir, f"{name}.txt")
            if os.path.exists(filename):
                os.remove(filename)
            self._load_templates()
            QMessageBox.information(self, "Usunięto", f"Usunięto szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się usunąć szablonu: {e}")


class OpenRouterTranslationSettingsDialog(QDialog):
    """Ustawienia tłumaczenia dla OpenRouter: klucz API + model + prompt + szablony."""
    def __init__(self, parent=None, current_key="", current_prompt="", current_model="google/gemini-2.5-flash", current_translation_segment_batch_size=250):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Tłumaczenia (OpenRouter API)")
        self.layout = QGridLayout(self)
        original_prompt = current_prompt or ""

        self.layout.addWidget(QLabel("Klucz API OpenRouter:"), 0, 0)
        self.key_input = QLineEdit(current_key)
        self.key_input.setPlaceholderText("Wklej klucz API OpenRouter...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.key_input, 0, 1, 1, 3)

        self.layout.addWidget(QLabel("Model OpenRouter:"), 1, 0)
        self.model_input = QComboBox()
        self.model_input.setEditable(False)
        self.model_input.addItem("Gemini 3 flash (preview)", userData="google/gemini-3-flash-preview")
        self.model_input.addItem("Gemini 2.5 flash", userData="google/gemini-2.5-flash")
        self.model_input.addItem("Gemini 3 pro", userData="google/gemini-3-pro-preview")
        self.model_input.addItem("Gemini 2.5 pro", userData="google/gemini-2.5-pro")
        self.model_input.addItem("GPT-5", userData="openai/gpt-5")
        self.model_input.addItem("GPT-5 mini", userData="openai/gpt-5-mini")
        model_to_set = current_model or "google/gemini-2.5-flash"
        idx = self.model_input.findData(model_to_set)
        if idx >= 0:
            self.model_input.setCurrentIndex(idx)
        else:
            self.model_input.setCurrentIndex(1)
        self.layout.addWidget(self.model_input, 1, 1, 1, 3)

        self.layout.addWidget(QLabel("Szablony promptów:"), 2, 0)
        self.template_combo = QComboBox()
        self.layout.addWidget(self.template_combo, 2, 1)

        self.save_template_btn = QPushButton("Zapisz szablon")
        self.save_template_btn.clicked.connect(self._save_template)
        self.layout.addWidget(self.save_template_btn, 2, 2)

        self.delete_template_btn = QPushButton("Usuń szablon")
        self.delete_template_btn.clicked.connect(self._delete_template)
        self.layout.addWidget(self.delete_template_btn, 2, 3)

        self.layout.addWidget(QLabel("Prompt (tłumaczenie):"), 3, 0)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Wprowadź prompt do tłumaczenia... Użyj {text}, {src_lang}, {tgt_lang}."
        )
        self.prompt_edit.setMinimumHeight(240)
        self.prompt_edit.setPlainText(current_prompt or "")
        self.layout.addWidget(self.prompt_edit, 4, 0, 1, 4)

        self.layout.addWidget(QLabel("Paczka segmentów SRT (tłumaczenie):"), 5, 0)
        self.translation_segment_batch_spin = QSpinBox()
        self.translation_segment_batch_spin.setRange(1, 1000)
        self.translation_segment_batch_spin.setValue(int(current_translation_segment_batch_size or 250))
        self.layout.addWidget(self.translation_segment_batch_spin, 5, 1, 1, 3)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 6, 0, 1, 4)

        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'translation')
        os.makedirs(self.prompts_dir, exist_ok=True)
        self.template_combo.currentTextChanged.connect(self._on_template_selected)
        self._load_templates()
        if isinstance(original_prompt, str) and original_prompt.strip():
            self.prompt_edit.setPlainText(original_prompt)

    def get_settings(self):
        model_id = self.model_input.currentData() or "google/gemini-2.5-flash"
        return self.key_input.text().strip(), self.prompt_edit.toPlainText().strip(), str(model_id).strip(), self.translation_segment_batch_spin.value()

    def _list_prompt_files(self):
        try:
            files = [f for f in os.listdir(self.prompts_dir) if os.path.isfile(os.path.join(self.prompts_dir, f)) and f.lower().endswith('.txt')]
            files.sort()
            return files
        except Exception:
            return []

    def _load_templates(self):
        current_prompt = self.prompt_edit.toPlainText().strip()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        files = self._list_prompt_files()
        for fn in files:
            self.template_combo.addItem(os.path.splitext(fn)[0])
        self.template_combo.blockSignals(False)

        if current_prompt:
            for fn in files:
                name = os.path.splitext(fn)[0]
                fp = os.path.join(self.prompts_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        if fh.read().strip() == current_prompt:
                            self.template_combo.blockSignals(True)
                            self.template_combo.setCurrentText(name)
                            self.template_combo.blockSignals(False)
                            break
                except Exception:
                    continue

    def _on_template_selected(self, name: str):
        if not name:
            return
        filename = os.path.join(self.prompts_dir, f"{name}.txt")
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as fh:
                    self.prompt_edit.setPlainText(fh.read())
        except Exception:
            pass

    def _sanitize_filename(self, name: str) -> str:
        import re
        s = name.strip().replace(' ', '_')
        s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
        if not s:
            s = 'prompt'
        return s

    def _save_template(self):
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text or len(prompt_text) < 20:
            QMessageBox.warning(self, "Za krótki prompt", "Prompt musi mieć co najmniej 20 znaków.")
            return
        name, ok = QInputDialog.getText(self, "Nazwa szablonu", "Podaj nazwę szablonu:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            safe = self._sanitize_filename(name)
            filename = os.path.join(self.prompts_dir, f"{safe}.txt")
            with open(filename, 'w', encoding='utf-8') as fh:
                fh.write(prompt_text)
            self._load_templates()
            try:
                self.template_combo.setCurrentText(name)
            except Exception:
                self.template_combo.setCurrentText(safe)
            QMessageBox.information(self, "Zapisano", f"Zapisano szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się zapisać szablonu: {e}")

    def _delete_template(self):
        name = self.template_combo.currentText()
        if not name:
            return
        ok = QMessageBox.question(self, "Usuń szablon", f"Czy na pewno chcesz usunąć szablon '{name}'?")
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            filename = os.path.join(self.prompts_dir, f"{name}.txt")
            if os.path.exists(filename):
                os.remove(filename)
            self._load_templates()
            QMessageBox.information(self, "Usunięto", f"Usunięto szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się usunąć szablonu: {e}")


class GeminiSummarySettingsDialog(QDialog):
    """Ustawienia streszczenia dla Gemini: klucz API + prompt + szablony (wspólne z Ollama)."""
    def __init__(self, parent=None, current_key="", current_prompt=""):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Streszczenia (Gemini API)")
        self.layout = QGridLayout(self)
        original_prompt = current_prompt or ""

        self.layout.addWidget(QLabel("Klucz API Gemini:"), 0, 0)
        self.key_input = QLineEdit(current_key)
        self.key_input.setPlaceholderText("Wklej klucz API Gemini...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.key_input, 0, 1, 1, 3)

        self.layout.addWidget(QLabel("Szablony promptów:"), 1, 0)
        self.template_combo = QComboBox()
        self.layout.addWidget(self.template_combo, 1, 1)

        self.save_template_btn = QPushButton("Zapisz szablon")
        self.save_template_btn.clicked.connect(self._save_template)
        self.layout.addWidget(self.save_template_btn, 1, 2)

        self.delete_template_btn = QPushButton("Usuń szablon")
        self.delete_template_btn.clicked.connect(self._delete_template)
        self.layout.addWidget(self.delete_template_btn, 1, 3)

        self.layout.addWidget(QLabel("Prompt (streszczenie):"), 2, 0)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Wprowadź prompt do streszczenia dla Gemini... Użyj {text}, {language_name}, {language_code}.")
        self.prompt_edit.setMinimumHeight(240)
        self.prompt_edit.setPlainText(current_prompt or "")
        self.layout.addWidget(self.prompt_edit, 3, 0, 1, 4)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 4, 0, 1, 4)

        # wspólny katalog szablonów dla sekcji streszczenie (Ollama + Gemini)
        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'summary')
        os.makedirs(self.prompts_dir, exist_ok=True)
        self.template_combo.currentTextChanged.connect(self._on_template_selected)
        self._load_templates()
        # Keep current prompt from settings; avoid auto-overwrite by first template.
        if isinstance(original_prompt, str) and original_prompt.strip():
            self.prompt_edit.setPlainText(original_prompt)

    def get_settings(self):
        return self.key_input.text().strip(), self.prompt_edit.toPlainText().strip()

    def _list_prompt_files(self):
        try:
            files = [f for f in os.listdir(self.prompts_dir) if os.path.isfile(os.path.join(self.prompts_dir, f)) and f.lower().endswith('.txt')]
            files.sort()
            return files
        except Exception:
            return []

    def _load_templates(self):
        current_prompt = self.prompt_edit.toPlainText().strip()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        files = self._list_prompt_files()
        for fn in files:
            self.template_combo.addItem(os.path.splitext(fn)[0])
        self.template_combo.blockSignals(False)

        # Select matching template name (without loading/overwriting prompt).
        if current_prompt:
            for fn in files:
                name = os.path.splitext(fn)[0]
                fp = os.path.join(self.prompts_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        if fh.read().strip() == current_prompt:
                            self.template_combo.blockSignals(True)
                            self.template_combo.setCurrentText(name)
                            self.template_combo.blockSignals(False)
                            break
                except Exception:
                    continue

    def _on_template_selected(self, name: str):
        if not name:
            return
        filename = os.path.join(self.prompts_dir, f"{name}.txt")
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as fh:
                    self.prompt_edit.setPlainText(fh.read())
        except Exception:
            pass

    def _sanitize_filename(self, name: str) -> str:
        import re
        s = name.strip().replace(' ', '_')
        s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
        if not s:
            s = 'prompt'
        return s

    def _save_template(self):
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text or len(prompt_text) < 20:
            QMessageBox.warning(self, "Za krótki prompt", "Prompt musi mieć co najmniej 20 znaków.")
            return
        name, ok = QInputDialog.getText(self, "Nazwa szablonu", "Podaj nazwę szablonu:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            safe = self._sanitize_filename(name)
            filename = os.path.join(self.prompts_dir, f"{safe}.txt")
            with open(filename, 'w', encoding='utf-8') as fh:
                fh.write(prompt_text)
            self._load_templates()
            try:
                self.template_combo.setCurrentText(name)
            except Exception:
                self.template_combo.setCurrentText(safe)
            QMessageBox.information(self, "Zapisano", f"Zapisano szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się zapisać szablonu: {e}")

    def _delete_template(self):
        name = self.template_combo.currentText()
        if not name:
            return
        ok = QMessageBox.question(self, "Usuń szablon", f"Czy na pewno chcesz usunąć szablon '{name}'?")
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            filename = os.path.join(self.prompts_dir, f"{name}.txt")
            if os.path.exists(filename):
                os.remove(filename)
            self._load_templates()
            QMessageBox.information(self, "Usunięto", f"Usunięto szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się usunąć szablonu: {e}")


class DiarizationDialog(QDialog):
    def __init__(self, parent=None, current_hf_token="", current_num_speakers=0):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Diaryzacji")
        
        self.layout = QGridLayout(self)
        
        self.layout.addWidget(QLabel("Token Hugging Face:"), 0, 0)
        self.hf_token_input = QLineEdit(current_hf_token)
        self.hf_token_input.setPlaceholderText("Wklej swój token dostępu...")
        self.hf_token_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.hf_token_input, 0, 1)

        self.num_speakers_label = QLabel("Liczba mówców (0 = auto):")
        self.num_speakers_label.setToolTip("Ustaw 0, aby model automatycznie wykrył liczbę mówców.")
        self.layout.addWidget(self.num_speakers_label, 1, 0)

        self.num_speakers_spin = QSpinBox()
        self.num_speakers_spin.setMinimum(0)
        self.num_speakers_spin.setValue(current_num_speakers)
        self.layout.addWidget(self.num_speakers_spin, 1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 2, 0, 1, 2)
        
    def get_settings(self):
        """Zwraca wprowadzony token i liczbę mówców."""
        return {
            "hf_token": self.hf_token_input.text().strip(),
            "num_speakers": self.num_speakers_spin.value()
        }


class GeminiSettingsDialog(QDialog):
    """Simple dialog to enter/store Gemini API key (no model choice)."""
    def __init__(self, parent=None, current_key=""):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia Gemini (API)")
        self.layout = QGridLayout(self)
        self.layout.addWidget(QLabel("Klucz API Gemini:"), 0, 0)
        self.key_input = QLineEdit(current_key)
        self.key_input.setPlaceholderText("Wklej klucz API Gemini...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.key_input, 0, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons, 1, 0, 1, 2)

    def get_key(self):
        return self.key_input.text().strip()
