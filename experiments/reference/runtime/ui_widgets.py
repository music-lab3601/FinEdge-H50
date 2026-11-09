"""Shared desktop palette and compact, readable value cards."""
import tkinter as tk

BG = '#0c1423'
PANEL = '#121f32'
CARD = '#192a40'
TEXT = '#edf3fc'
MUTED = '#9baec7'
ACCENT = '#74adff'
UP, DOWN, FLAT = '#ff8795', '#58d9ae', '#c5d3e6'
SELECT, REALIZED = '#ffd07a', '#69dadd'


def direction(cls):
    return ('↓ 看跌', '→ 持平', '↑ 看涨')[cls] if cls is not None else '未启用'


def direction_color(cls):
    return (DOWN, FLAT, UP)[cls] if cls is not None else MUTED


class ValueCard(tk.Frame):
    def __init__(self, parent, title, value='—', note='', size=22, compact=False):
        super().__init__(parent, bg=CARD, padx=12, pady=5 if compact else 9,
                         highlightbackground='#263b56', highlightthickness=1)
        self.title_label = tk.Label(self, text=title, bg=CARD, fg=MUTED,
                                    font=('Microsoft YaHei UI', 8 if compact else 9), anchor='w')
        self.title_label.pack(fill='x')
        self.value_label = tk.Label(self, text=value, bg=CARD, fg=TEXT,
                                    font=('Microsoft YaHei UI', size, 'bold'), anchor='w')
        self.value_label.pack(fill='x', pady=(2, 1))
        self.note_label = tk.Label(self, text=note, bg=CARD, fg=MUTED,
                                   font=('Microsoft YaHei UI', 8 if compact else 9), anchor='w', justify='left')
        self.note_label.pack(fill='x')
        self.bind('<Configure>', self._resize)

    def _resize(self, event):
        self.note_label.configure(wraplength=max(120, event.width-26))

    def set(self, value, note='', color=TEXT):
        self.value_label.configure(text=value, fg=color)
        self.note_label.configure(text=note)
