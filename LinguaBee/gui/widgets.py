from PySide6.QtWidgets import (
    QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QGroupBox,
    QGridLayout, QSizePolicy
)

class SourceGroup(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Źródło", parent)
        self.layout = QGridLayout(self)
        # consistent spacing/margins for all groups
        self.layout.setVerticalSpacing(8)
        self.layout.setHorizontalSpacing(8)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.url_entry = QLineEdit()
        self.url_entry.setObjectName("SourceUrlEntry")
        self.url_entry.setMinimumHeight(30)
        self.url_entry.setPlaceholderText("WSTAW LINK DO YOUTUBE")
        self.file_btn = QPushButton("WYBIERZ PLIK LOKALNY")
        self.file_btn.setObjectName("SourceFileButton")
        self.file_btn.setMinimumHeight(32)
        self.file_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout.addWidget(self.url_entry, 0, 0)
        self.layout.addWidget(self.file_btn, 2, 0)
        self.layout.setRowStretch(1, 1)
        self.layout.setColumnMinimumWidth(0, 96)
        self.layout.setColumnStretch(0, 1)

class TranscriptionGroup(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Transkrypcja", parent)
        self.layout = QGridLayout(self)
        # uniform spacing & margins so rows align visually with other groups
        self.layout.setVerticalSpacing(6)
        self.layout.setHorizontalSpacing(8)
        # keep symmetric top/bottom margins so the group padding is balanced
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Brak", "Whisper (lokalny)"])
        self.model_combo.setCurrentText("Whisper (lokalny)")
        self.layout.addWidget(QLabel("Model\ntranskrypcji:"), 0, 0)
        self.layout.addWidget(self.model_combo, 0, 1)
        self.languages = {
            "Angielski": "en",
            "Chiński (mandaryński)": "zh",
            "Hindi": "hi",
            "Hiszpański": "es",
            "Francuski": "fr",
            "Arabski (standardowy)": "ar",
            "Bengalski": "bn",
            "Rosyjski": "ru",
            "Portugalski": "pt",
            "Urdu": "ur",
            "Indonezyjski": "id",
            "Niemiecki": "de",
            "Japoński": "ja",
            "Turecki": "tr",
            "Koreański": "ko",
            "Wietnamski": "vi",
            "Włoski": "it",
            "Polski": "pl",
            "Ukraiński": "uk",
            "Holenderski": "nl",
            "Perski (Farsi)": "fa",
            "Szwedzki": "sv",
            "Rumuński": "ro",
            "Grecki": "el",
            "Hebrajski": "he"
        }
        language_names = sorted(self.languages.keys(), key=str.casefold)
        self.src_lang_combo = QComboBox()
        self.src_lang_combo.addItems(["auto"] + language_names)
        self.layout.addWidget(QLabel("Język\nźródłowy:"), 1, 0)
        self.layout.addWidget(self.src_lang_combo, 1, 1)

        # Correction / post-editing option (Korekta)
        self.correction_combo = QComboBox()
        self.correction_combo.addItems(["Brak", "Ollama (lokalny)", "Gemini (API)", "OpenRouter (API)"])
        self.correction_combo.setCurrentText("Brak")
        self.layout.addWidget(QLabel("Korekta:"), 2, 0)
        self.layout.addWidget(self.correction_combo, 2, 1)

        # Use only 3 rows and slightly smaller row height so the groupbox hugs the last widget
        for r in range(3):
            self.layout.setRowMinimumHeight(r, 22)
        self.layout.setColumnStretch(1, 1)

class TranslationGroup(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Tłumaczenie", parent)
        self.layout = QGridLayout(self)
        # uniform spacing & margins so rows align visually with other groups
        self.layout.setVerticalSpacing(6)
        self.layout.setHorizontalSpacing(8)
        # keep symmetric top/bottom margins so the group padding is balanced
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.translation_combo = QComboBox()
        # Removed OpenAI and Gemini options per user request
        self.translation_combo.addItems(["Brak", "NLLB (lokalny)", "Helsinki (lokalny)", "Ollama (lokalny)", "OpenRouter (API)"])
        self.translation_combo.setCurrentText("NLLB (lokalny)")
        self.languages = {
            "Angielski": "en",
            "Chiński (mandaryński)": "zh",
            "Hindi": "hi",
            "Hiszpański": "es",
            "Francuski": "fr",
            "Arabski (standardowy)": "ar",
            "Bengalski": "bn",
            "Rosyjski": "ru",
            "Portugalski": "pt",
            "Urdu": "ur",
            "Indonezyjski": "id",
            "Niemiecki": "de",
            "Japoński": "ja",
            "Turecki": "tr",
            "Koreański": "ko",
            "Wietnamski": "vi",
            "Włoski": "it",
            "Polski": "pl",
            "Ukraiński": "uk",
            "Holenderski": "nl",
            "Perski (Farsi)": "fa",
            "Szwedzki": "sv",
            "Rumuński": "ro",
            "Grecki": "el",
            "Hebrajski": "he"
        }
        language_names = sorted(self.languages.keys(), key=str.casefold)

        self.translation_src_lang_combo = QComboBox()
        self.translation_src_lang_combo.addItems(["auto"] + language_names)
        self.translation_src_lang_combo.setCurrentText("auto")
        self.translation_src_lang_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.tgt_lang_combo = QComboBox()
        self.tgt_lang_combo.addItems(language_names)
        self.tgt_lang_combo.setCurrentText("Polski")
        self.tgt_lang_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout.addWidget(QLabel("Model:"), 0, 0)
        self.layout.addWidget(self.translation_combo, 0, 1)
        self.layout.addWidget(QLabel("Język\nźródłowy:"), 1, 0)
        self.layout.addWidget(self.translation_src_lang_combo, 1, 1)
        self.layout.addWidget(QLabel("Język\ndocelowy:"), 2, 0)
        self.layout.addWidget(self.tgt_lang_combo, 2, 1)
        # Match TranscriptionGroup: 3 rows with smaller minimum height
        for r in range(3):
            self.layout.setRowMinimumHeight(r, 22)
        self.layout.setColumnStretch(1, 1)

class SummaryGroup(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Streszczenie", parent)
        self.layout = QGridLayout(self)
        # uniform spacing & margins so rows align visually with other groups
        self.layout.setVerticalSpacing(6)
        self.layout.setHorizontalSpacing(8)
        # keep symmetric top/bottom margins so the group padding is balanced
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.summary_combo = QComboBox()
        self.summary_combo.addItems(["Brak", "Ollama (lokalny)", "Gemini (API)", "OpenRouter (API)", "BART (lokalny)"])
        self.summary_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.languages = {
            "Angielski": "en",
            "Chiński (mandaryński)": "zh",
            "Hindi": "hi",
            "Hiszpański": "es",
            "Francuski": "fr",
            "Arabski (standardowy)": "ar",
            "Bengalski": "bn",
            "Rosyjski": "ru",
            "Portugalski": "pt",
            "Urdu": "ur",
            "Indonezyjski": "id",
            "Niemiecki": "de",
            "Japoński": "ja",
            "Turecki": "tr",
            "Koreański": "ko",
            "Wietnamski": "vi",
            "Włoski": "it",
            "Polski": "pl",
            "Ukraiński": "uk",
            "Holenderski": "nl",
            "Perski (Farsi)": "fa",
            "Szwedzki": "sv",
            "Rumuński": "ro",
            "Grecki": "el",
            "Hebrajski": "he"
        }
        language_names = sorted(self.languages.keys(), key=str.casefold)

        self.summary_lang_combo = QComboBox()
        self.summary_lang_combo.addItems(language_names)
        self.summary_lang_combo.setCurrentText("Polski")
        self.summary_lang_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout.addWidget(QLabel("Model:"), 0, 0)
        self.layout.addWidget(self.summary_combo, 0, 1)
        self.layout.addWidget(QLabel("Język\nstreszczenia:"), 1, 0)
        self.layout.addWidget(self.summary_lang_combo, 1, 1)
        # Summary group is smaller: 2 rows with the same reduced height
        for r in range(2):
            self.layout.setRowMinimumHeight(r, 22)
        self.layout.setColumnStretch(1, 1)

class FormatsGroup(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Formaty wyjściowe", parent)
        self.layout = QGridLayout(self)
        # consistent spacing for the formats grid as well
        self.layout.setVerticalSpacing(6)
        self.layout.setHorizontalSpacing(8)
        self.layout.setContentsMargins(6, 6, 6, 6)

        self.txt_cb_orig = QCheckBox("TXT")
        self.txt_cb_orig.setChecked(True)
        self.docx_cb_orig = QCheckBox("DOCX")
        self.docx_cb_orig.setChecked(True)
        self.html_cb_orig = QCheckBox("HTML")
        self.srt_cb_orig = QCheckBox("SRT")
        self.srt_cb_orig.setChecked(True)

        self.txt_cb_trans = QCheckBox("TXT")
        self.docx_cb_trans = QCheckBox("DOCX")
        self.docx_cb_trans.setChecked(True)
        self.html_cb_trans = QCheckBox("HTML")
        self.srt_cb_trans = QCheckBox("SRT")
        self.srt_cb_trans.setChecked(True)

        self.txt_cb_summ = QCheckBox("TXT")
        self.txt_cb_summ.setChecked(True)
        self.docx_cb_summ = QCheckBox("DOCX")
        self.docx_cb_summ.setChecked(True)
        self.html_cb_summ = QCheckBox("HTML")

        self.original_checkboxes = [self.txt_cb_orig, self.docx_cb_orig, self.html_cb_orig, self.srt_cb_orig]
        self.translated_checkboxes = [self.txt_cb_trans, self.docx_cb_trans, self.html_cb_trans, self.srt_cb_trans]
        self.summary_checkboxes = [self.txt_cb_summ, self.docx_cb_summ, self.html_cb_summ]

        self.layout.addWidget(QLabel("Transkrypcja"), 0, 0)
        self.layout.addWidget(QLabel("Tłumaczenie"), 0, 1)
        self.layout.addWidget(QLabel("Streszczenie"), 0, 2)
        self.layout.addWidget(self.txt_cb_orig, 1, 0)
        self.layout.addWidget(self.txt_cb_trans, 1, 1)
        self.layout.addWidget(self.txt_cb_summ, 1, 2)
        self.layout.addWidget(self.docx_cb_orig, 2, 0)
        self.layout.addWidget(self.docx_cb_trans, 2, 1)
        self.layout.addWidget(self.docx_cb_summ, 2, 2)
        self.layout.addWidget(self.html_cb_orig, 3, 0)
        self.layout.addWidget(self.html_cb_trans, 3, 1)
        self.layout.addWidget(self.html_cb_summ, 3, 2)
        self.layout.addWidget(self.srt_cb_orig, 4, 0)
        self.layout.addWidget(self.srt_cb_trans, 4, 1)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
