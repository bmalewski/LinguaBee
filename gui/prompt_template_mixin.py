"""Mixin obs\u0142uguj\u0105cy szablony prompt\u00f3w (zapisywane jako pliki .txt).

Wcze\u015bniej identyczny zestaw metod (`_list_prompt_files`, `_load_templates`,
`_on_template_selected`, `_sanitize_filename`, `_save_template`,
`_delete_template`) by\u0142 kopiowany w wielu dialogach w `gui/dialogs.py`. Ten mixin
centralizuje t\u0119 logik\u0119.

Wymagania wzgl\u0119dem klasy korzystaj\u0105cej z mixinu:
- atrybut `self.prompts_dir` (\u015bcie\u017cka do katalogu z szablonami),
- atrybut `self.template_combo` (QComboBox z nazwami szablon\u00f3w),
- atrybut `self.prompt_edit` (QTextEdit z tre\u015bci\u0105 promptu),
- domy\u015blna nazwa pliku przy pustej sanityzacji mo\u017ce by\u0107 nadpisana przez
  `self.default_prompt_filename`.
"""

import os
import re

from PySide6.QtWidgets import QMessageBox, QInputDialog


class PromptTemplateMixin:
    #: Domy\u015blna nazwa pliku u\u017cywana, gdy sanityzacja zwr\u00f3ci pusty ci\u0105g.
    default_prompt_filename = "prompt"

    def _list_prompt_files(self):
        try:
            files = [
                f for f in os.listdir(self.prompts_dir)
                if os.path.isfile(os.path.join(self.prompts_dir, f)) and f.lower().endswith(".txt")
            ]
            files.sort()
            return files
        except Exception:
            return []

    def _load_templates(self):
        current_prompt = self.prompt_edit.toPlainText().strip()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        files = self._list_prompt_files()
        # Niekt\u00f3re dialogi przechowuj\u0105 list\u0119 nazw szablon\u00f3w; utrzymujemy j\u0105 dla zgodno\u015bci.
        self.template_names = []
        for fn in files:
            name = os.path.splitext(fn)[0]
            self.template_combo.addItem(name)
            self.template_names.append(name)
        self.template_combo.blockSignals(False)

        # Zaznacz pasuj\u0105cy szablon (bez nadpisywania bie\u017c\u0105cego promptu).
        if current_prompt:
            for fn in files:
                name = os.path.splitext(fn)[0]
                fp = os.path.join(self.prompts_dir, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
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
                with open(filename, "r", encoding="utf-8") as fh:
                    self.prompt_edit.setPlainText(fh.read())
        except Exception:
            pass

    def _sanitize_filename(self, name: str) -> str:
        s = name.strip().replace(" ", "_")
        s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
        return s or self.default_prompt_filename

    def _save_template(self):
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text or len(prompt_text) < 20:
            QMessageBox.warning(self, "Za kr\u00f3tki prompt", "Prompt musi mie\u0107 co najmniej 20 znak\u00f3w.")
            return
        name, ok = QInputDialog.getText(self, "Nazwa szablonu", "Podaj nazw\u0119 szablonu:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            safe = self._sanitize_filename(name)
            filename = os.path.join(self.prompts_dir, f"{safe}.txt")
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(prompt_text)
            self._load_templates()
            try:
                self.template_combo.setCurrentText(name)
            except Exception:
                self.template_combo.setCurrentText(safe)
            QMessageBox.information(self, "Zapisano", f"Zapisano szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "B\u0142\u0105d", f"Nie uda\u0142o si\u0119 zapisa\u0107 szablonu: {e}")

    def _delete_template(self):
        name = self.template_combo.currentText()
        if not name:
            return
        ok = QMessageBox.question(self, "Usu\u0144 szablon", f"Czy na pewno chcesz usun\u0105\u0107 szablon '{name}'?")
        if ok != QMessageBox.StandardButton.Yes:
            return
        try:
            filename = os.path.join(self.prompts_dir, f"{name}.txt")
            if os.path.exists(filename):
                os.remove(filename)
            self._load_templates()
            QMessageBox.information(self, "Usuni\u0119to", f"Usuni\u0119to szablon '{name}'.")
        except Exception as e:
            QMessageBox.warning(self, "B\u0142\u0105d", f"Nie uda\u0142o si\u0119 usun\u0105\u0107 szablonu: {e}")
