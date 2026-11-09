"""Independent product dashboard; one FPGA/NPU prediction path only."""
from collections import OrderedDict, deque
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .fpga_streaming import LatestDisplay
from .market import CandleSeries, INTERVALS, exchange_time
from .market_view import MarketView
from .paths import CLASS_CN
from .showcase_replay import BACKEND, export_showcase_csv, run_showcase
from .showcase_data import DEFAULT_DATASET, LABELS
from .delivery import read_json
from .ui_widgets import BG, PANEL, CARD, TEXT, MUTED, ACCENT, SELECT, ValueCard, direction, direction_color

PRODUCT = 'FinEdge 智能盘口分析系统'


class ShowcaseMarketView(MarketView):
    """Use existing chart geometry, with independently constructed NPU-only UI."""
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.series = {label: CandleSeries(ns, 120) for label, ns in INTERVALS.items()}
        self.observations = deque(maxlen=20000)
        self.predictions = OrderedDict()
        self.selected = self.latest_resolved = self.latest_prediction = None
        self.dirty = True; self.hitboxes = []; self.on_follow = lambda: None
        self.interval = tk.StringVar(value='100毫秒')
        self.chart_kind = tk.StringVar(value='K线')
        self.visible_bars = tk.StringVar(value='60')
        self.market_text = tk.StringVar(value='AAPL · 等待历史行情')
        self.price_text = tk.StringVar(value='— USD')
        self.market_clock = tk.StringVar(value='纽约交易所时间')
        self.accuracy_text = tk.StringVar(value='等待未来结果')
        self.selection_text = tk.StringVar(value='等待预测')
        self.review_title = tk.StringVar(value='预测复盘 · 未来50条盘口到达后核对方向')
        self.hover_text = tk.StringVar(value='K线由最优买卖价的中间价聚合，非成交价K线；虚线框表示当前时间桶尚未结束。')
        self.score_npu = tk.StringVar(value='方向命中 —')
        self.score_count = tk.StringVar(value='已核对 0 · 等待 0')
        self.columnconfigure(0, weight=1); self.columnconfigure(1, minsize=235)
        self.rowconfigure(0, weight=1, minsize=245)
        chart = ttk.Frame(self); chart.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
        quote = ttk.Frame(chart); quote.pack(fill='x')
        ttk.Label(quote, text='盘口中间价 · 历史行情', style='Section.TLabel').pack(side='left')
        ttk.Label(quote, textvariable=self.price_text, style='Price.TLabel').pack(side='right')
        ttk.Label(chart, textvariable=self.market_text, foreground=MUTED).pack(anchor='w')
        controls = ttk.Frame(chart); controls.pack(fill='x', pady=5)
        for variable, values, width in ((self.chart_kind, ('K线', '中间价走势'), 12),
                (self.interval, tuple(INTERVALS), 9), (self.visible_bars, ('30', '60', '120'), 5)):
            box = ttk.Combobox(controls, textvariable=variable, values=values, width=width, state='readonly')
            box.pack(side='left', padx=(0, 6))
            box.bind('<<ComboboxSelected>>', lambda _: self.invalidate())
        ttk.Label(controls, text='个时间桶 · 仅显示已回放行情', foreground=MUTED).pack(side='left')
        self.canvas = tk.Canvas(chart, bg=PANEL, height=245, highlightthickness=1, highlightbackground='#263b56')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _: self.invalidate()); self.canvas.bind('<Motion>', self.hover)
        ttk.Label(chart, text='A 金色：发出预测     B 青色：50条后的实际价格     红涨 / 绿跌',
                  foreground=MUTED).pack(anchor='w', pady=(4, 0))
        ttk.Label(chart, textvariable=self.hover_text, foreground=MUTED, wraplength=850).pack(anchor='w')
        sidebar = ttk.Frame(self); sidebar.grid(row=0, column=1, sticky='nsew')
        ttk.Label(sidebar, text='板载智能预测', style='Section.TLabel').pack(anchor='w')
        ttk.Label(sidebar, text='目标：未来第50条盘口的方向', foreground=MUTED).pack(anchor='w', pady=(3, 8))
        self.live_npu = ValueCard(sidebar, 'NPU 预测方向', '等待预测', '满100条盘口后生成首窗', size=22, compact=True)
        self.live_npu.pack(fill='x')
        self.logits_text = tk.StringVar(value='跌 / 平 / 涨分数：—')
        ttk.Label(sidebar, textvariable=self.logits_text, wraplength=225, foreground=MUTED).pack(anchor='w', pady=6)
        ttk.Separator(sidebar).pack(fill='x', pady=3)
        ttk.Label(sidebar, textvariable=self.score_npu, wraplength=225).pack(anchor='w', pady=2)
        review = ttk.Frame(self); review.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        header = ttk.Frame(review); header.pack(fill='x', pady=(0, 5))
        ttk.Label(header, textvariable=self.review_title, style='Section.TLabel').pack(side='left')
        ttk.Button(header, text='跟随最新结果', command=self.follow).pack(side='right')
        cards = ttk.Frame(review); cards.pack(fill='x')
        self.origin_card = ValueCard(cards, 'A · 预测时的真实价格', '等待行情', '预测起点', size=20)
        self.forecast_card = ValueCard(cards, 'NPU 当时的判断', '等待预测', '判断方向，不预测具体价格', size=20)
        self.actual_card = ValueCard(cards, 'B · 50条后的实际结果', '尚未到达', '未来到达后才判定对错', size=20)
        for index, card in enumerate((self.origin_card, self.forecast_card, self.actual_card)):
            cards.columnconfigure(index, weight=1, uniform='review')
            card.grid(row=0, column=index, sticky='nsew', padx=(0 if index == 0 else 5, 0 if index == 2 else 5))

    def consume(self, item):
        if item['type'] == 'prediction':
            result = item['outputs'][BACKEND]
            cls = result['class_index']
            self.live_npu.set(direction(cls), f"窗口 #{item['window_id']} · FPGA 组窗输入", direction_color(cls))
            self.logits_text.set('跌 / 平 / 涨分数（非概率）\n'+' / '.join(f'{x:.4f}' for x in result['logits']))
            self.predictions[item['window_id']] = item
            self.latest_prediction = item['window_id']
            while len(self.predictions) > 150: self.predictions.popitem(last=False)
            self.update_evaluation(item.get('evaluation', {}))
        self.dirty = True

    def update_evaluation(self, summary):
        metric = summary.get('backends', {}).get(BACKEND)
        self.score_npu.set(f"方向命中 {metric['correct']}/{metric['count']} · {metric['accuracy']:.1%}" if metric else '方向命中 —')
        self.score_count.set(f"已核对 {summary.get('resolved', 0)} · 等待 {summary.get('pending', 0)} · 未兑现 {summary.get('unresolved', 0)}")
        self.accuracy_text.set(self.score_npu.get()+' · '+self.score_count.get())

    def clear(self, new_run=False):
        for series in self.series.values(): series.bars.clear()
        self.observations.clear(); self.predictions.clear()
        self.selected = self.latest_resolved = self.latest_prediction = None
        self.live_npu.set('等待预测', '满100条盘口后生成首窗', MUTED)
        self.update_evaluation({}); self.dirty = True

    def review_prediction(self):
        key = self.selected or self.latest_resolved or self.latest_prediction
        row = self.predictions.get(key)
        if not row: return None
        cls = row['outputs'][BACKEND]['class_index']; outcome = row.get('outcome', {})
        self.review_title.set(f"预测复盘 · 窗口 #{key} · "+('手动选中' if self.selected else '自动跟随'))
        self.origin_card.set(f"${row['origin_mid2']/20000:.5f}",
            exchange_time(row['last']['timestamp_ns_since_exchange_midnight'])+' 纽约 · 预测起点 A', SELECT)
        self.forecast_card.set(direction(cls), '未来第50条有效盘口的中间价方向', direction_color(cls))
        if outcome.get('status') == 'resolved':
            actual = outcome['actual_class']; correct = outcome['matches'][BACKEND]
            change = (outcome['actual_mid2']-row['origin_mid2'])/20000
            self.actual_card.set(('↓ 实际下跌', '→ 实际持平', '↑ 实际上涨')[actual],
                f"${outcome['actual_mid2']/20000:.5f}（{change:+.5f}） · "+('预测命中' if correct else '预测未命中'), direction_color(actual))
        elif outcome:
            self.actual_card.set('未兑现', '片段结束或停止时，未来不足50条\n不计入命中率', MUTED)
        else: self.actual_card.set('等待未来50条', '真实结果尚未到达，暂不判定对错', MUTED)
        self.selection_text.set(f'窗口 #{key} · NPU {CLASS_CN[cls]}')
        return row


class ShowcaseApp(tk.Tk):
    def __init__(self, session_path, *, runner=run_showcase):
        super().__init__()
        self.session_path = session_path; self.runner = runner
        self.title(PRODUCT+' · FPGA ＋ NPU'); self.geometry('1320x920'); self.minsize(1100, 830)
        self.configure(bg=BG)
        self.mailbox = LatestDisplay(); self.messages = queue.Queue()
        self.paused = threading.Event(); self.cancelled = threading.Event()
        self.thread = self.export_thread = None
        self.consumed = self.closing = False; self.run_directory = None; self.snapshot = None
        self.rate_target = 20.; self.rate = tk.StringVar(value='20')
        self.status = tk.StringVar(value='就绪 · 开始后等待板端初始化和预热')
        self.progress = tk.StringVar(value='有效盘口 0 · 预测窗口 0')
        self.note = tk.StringVar(value='原始行情 → 电脑建簿与归一化 → FPGA组窗 → 板载NPU → 方向预测与复盘')
        self.data_note = tk.StringVar(value='行情：'+LABELS[DEFAULT_DATASET]+(
            ' · 尚未与队友原验证集逐条对齐' if DEFAULT_DATASET=='databento20251215' else ''))
        contract_path=Path(session_path).parent/'showcase_contract.json'
        if contract_path.is_file():
            contract=read_json(contract_path);data=contract.get('data_source',{})
            if contract.get('schema')=='native-showcase-v2':
                self.note.set('原始行情 → 运行时建簿与归一化 → FPGA共享窗口 → 驻留NPU → 预测复盘')
                self.data_note.set('模型：'+contract['model_label'])
            if data.get('label'):
                self.data_note.set('行情：'+data['label']+(
                    ' · 尚未与队友原验证集逐条对齐' if data.get('dataset_id')=='databento20251215' else ''))
                if contract.get('schema')=='native-showcase-v2':self.data_note.set(self.data_note.get()+'\n模型：'+contract['model_label'])
        self._style(); self._build()
        self.protocol('WM_DELETE_WINDOW', self.close_app); self.after(100, self.poll)

    def _style(self):
        style = ttk.Style(self); style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=TEXT, font=('Microsoft YaHei UI', 9))
        style.configure('Section.TLabel', font=('Microsoft YaHei UI', 11, 'bold'))
        style.configure('Price.TLabel', font=('Consolas', 22, 'bold'))
        style.configure('TButton', background=CARD, foreground=TEXT, padding=(10, 5), borderwidth=0)
        style.map('TButton', background=[('active', '#29476b')], foreground=[('disabled', MUTED)])
        style.configure('TCombobox', fieldbackground=PANEL, background=CARD, foreground=TEXT, padding=4)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', TEXT)])
        style.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=25, borderwidth=0, font=('Microsoft YaHei UI', 9))
        style.configure('Treeview.Heading', background=CARD, foreground=MUTED, padding=5)
        style.map('Treeview', background=[('selected', '#2b4e77')])
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=MUTED, padding=(14, 6))
        style.map('TNotebook.Tab', background=[('selected', CARD)], foreground=[('selected', TEXT)])

    def _build(self):
        box = ttk.Frame(self, padding=(20, 12)); box.pack(fill='both', expand=True)
        box.columnconfigure(0, weight=1); box.rowconfigure(4, weight=1)
        header = ttk.Frame(box); header.grid(row=0, column=0, sticky='ew')
        tk.Label(header, text='FinEdge', bg=BG, fg=TEXT, font=('Segoe UI', 25, 'bold')).pack(side='left')
        ttk.Label(header, text='智能盘口分析系统', font=('Microsoft YaHei UI', 14)).pack(side='left', padx=14)
        ttk.Label(header, text='FPGA ＋ NPU', foreground=ACCENT, font=('Segoe UI', 13, 'bold')).pack(side='right')
        ttk.Label(box, textvariable=self.note, foreground=MUTED).grid(row=1, column=0, sticky='w', pady=(3, 10))
        controls = ttk.Frame(box); controls.grid(row=2, column=0, sticky='ew')
        self.start_button = ttk.Button(controls, text='开始回放', command=self.start); self.start_button.pack(side='left')
        self.pause_button = ttk.Button(controls, text='暂停', command=self.pause, state='disabled'); self.pause_button.pack(side='left', padx=6)
        self.stop_button = ttk.Button(controls, text='停止并保存', command=self.stop, state='disabled'); self.stop_button.pack(side='left')
        self.export_button = ttk.Button(controls, text='导出预测 CSV', command=self.export_log); self.export_button.pack(side='left', padx=6)
        ttk.Label(controls, text='回放速度（窗/秒）').pack(side='left', padx=(20, 6))
        speed = ttk.Combobox(controls, textvariable=self.rate, values=('10', '20', '50', '0'), state='readonly', width=6)
        speed.pack(side='left'); speed.bind('<<ComboboxSelected>>', self.set_rate)
        ttk.Label(controls, text='0 = 不限速', foreground=MUTED).pack(side='left', padx=6)
        ttk.Button(controls, text='界面说明', command=self.show_help).pack(side='right')
        metrics = ttk.Frame(box); metrics.grid(row=3, column=0, sticky='ew', pady=10)
        self.last_time = ValueCard(metrics, '最新 NPU 调用', '— ms', '等待正式预测', size=22, compact=True)
        self.mean_time = ValueCard(metrics, '平均 NPU 调用', '— ms', '不含启动预热', size=22, compact=True)
        self.count_card = ValueCard(metrics, '已完成预测', '0', '每窗100条盘口 · 步长1', size=22, compact=True)
        self.board_time = ValueCard(metrics,'板内完整通路','— ms','FPGA＋NPU＋ARM交接',size=22,compact=True)
        self.roundtrip_time = ValueCard(metrics,'整批请求往返','— ms','包含板内计算与网络',size=22,compact=True)
        for i, card in enumerate((self.last_time, self.mean_time,self.board_time,self.roundtrip_time,self.count_card)):
            metrics.columnconfigure(i, weight=1, uniform='metric')
            card.grid(row=0, column=i, sticky='ew', padx=(0 if i == 0 else 5, 0 if i == 2 else 5))
        self.market_view = ShowcaseMarketView(box); self.market_view.grid(row=4, column=0, sticky='nsew')
        tabs = ttk.Notebook(box, height=115); tabs.grid(row=5, column=0, sticky='ew', pady=(10, 6))
        history_frame = ttk.Frame(tabs); book_frame = ttk.Frame(tabs)
        tabs.add(history_frame, text='预测记录 · 点击复盘'); tabs.add(book_frame, text='实时十档盘口')
        self.history = ttk.Treeview(history_frame, columns=('id', 'time', 'forecast', 'actual', 'verdict', 'npu_ms'), show='headings', height=3)
        for key, label, width in zip(self.history['columns'], ('窗口', '预测时刻（纽约）', 'NPU 预测', '50条后实际', '判定', 'NPU 调用 ms'), (70, 200, 150, 150, 130, 140)):
            self.history.heading(key, text=label); self.history.column(key, width=width, anchor='center')
        scroll = ttk.Scrollbar(history_frame, command=self.history.yview); scroll.pack(side='right', fill='y')
        self.history.configure(yscrollcommand=scroll.set); self.history.pack(fill='both', expand=True)
        self.history.bind('<<TreeviewSelect>>', self.select_prediction)
        self.market_view.on_follow = lambda: self.history.selection_remove(*self.history.selection())
        self.book = ttk.Treeview(book_frame, columns=('level', 'ask', 'askqty', 'bid', 'bidqty'), show='headings', height=3)
        for key, label in zip(self.book['columns'], ('档位', '卖价 USD', '卖量（股）', '买价 USD', '买量（股）')):
            self.book.heading(key, text=label); self.book.column(key, width=140, anchor='e')
        for i in range(10): self.book.insert('', 'end', iid=str(i), values=(i+1, '—', '—', '—', '—'))
        scroll = ttk.Scrollbar(book_frame, command=self.book.yview); scroll.pack(side='right', fill='y')
        self.book.configure(yscrollcommand=scroll.set); self.book.pack(fill='both', expand=True)
        footer=ttk.Frame(box);footer.grid(row=6,column=0,sticky='ew')
        ttk.Label(footer,textvariable=self.data_note,foreground=MUTED).pack(anchor='w')
        ttk.Label(footer, textvariable=self.progress, foreground=MUTED).pack(anchor='w')
        ttk.Label(box, textvariable=self.status, wraplength=1240).grid(row=7, column=0, sticky='w', pady=4)
        ttk.Label(box, text='NPU：启动、等待与输出读取；板内：新记录到预测包及槽位释放。批次往返按整批计，均不含界面绘制。',
                  foreground=MUTED).grid(row=8, column=0, sticky='w')

    def set_rate(self, event=None):
        from .demo_rate import validate_rate
        self.rate_target = validate_rate(float(self.rate.get()))

    def start(self):
        if self.consumed: return
        self.set_rate(); self.consumed = True
        self.start_button.configure(state='disabled'); self.pause_button.configure(state='normal'); self.stop_button.configure(state='normal')
        self.status.set('正在初始化板端服务；首次启动包含预热，请等待。')
        def work():
            try:
                result = self.runner(self.session_path, display=self.mailbox, paused=self.paused,
                    cancelled=self.cancelled, get_rate=lambda: self.rate_target)
                self.messages.put(('result', result))
            except Exception as exc: self.messages.put(('error', str(exc)))
        self.thread = threading.Thread(target=work, name='showcase-fpga-npu', daemon=False); self.thread.start()

    def pause(self):
        if not self.thread or not self.thread.is_alive() or self.cancelled.is_set(): return
        if self.paused.is_set():
            self.paused.clear(); self.pause_button.configure(text='暂停'); self.status.set('继续回放，沿用原会话。')
        else:
            self.paused.set(); self.pause_button.configure(text='继续'); self.status.set('正在等待当前批次完成后暂停…')

    def stop(self):
        self.cancelled.set(); self.paused.clear()
        self.pause_button.configure(state='disabled'); self.stop_button.configure(state='disabled')
        self.status.set('正在完成当前批次并保存结果，请保持网络和电源连接。')

    def apply_snapshot(self, snapshot):
        self.snapshot = snapshot
        if snapshot.get('directory'): self.run_directory = Path(snapshot['directory'])
        self.market_view.apply_snapshot(snapshot)
        stats = snapshot['stats']; count = stats.get('windows', 0)
        ns = stats.get('npu_call_last_ns')
        if ns is not None: self.last_time.set(f'{ns/1e6:.3f} ms', '板端正式预测调用')
        if count: self.mean_time.set(f"{stats['npu_call_total_ns']/count/1e6:.3f} ms", f'累计 {count:,} 窗，不含预热')
        hot=stats.get('board_pipeline_count',0)
        if hot:self.board_time.set(f"{stats['board_pipeline_total_ns']/hot/1e6:.3f} ms",f'平均 {hot:,} 个热窗，不含首窗初始化')
        rtt=stats.get('batch_roundtrip_last_ns')
        if rtt is not None:self.roundtrip_time.set(f'{rtt/1e6:.3f} ms','含板内计算与网络，可含多窗')
        evaluation = snapshot.get('evaluation', {})
        self.count_card.set(f'{count:,}', f"已核对 {evaluation.get('resolved', 0):,} · 等待 {evaluation.get('pending', 0):,} · 未兑现 {evaluation.get('unresolved', 0):,}")
        self.progress.set(f"有效盘口 {stats.get('records', 0):,} · 预测窗口 {count:,} / {snapshot.get('expected_windows', 0):,} · 历史行情单次回放")
        phase = snapshot.get('phase')
        if phase in ('paused', 'awaiting_continuation'):
            self.pause_button.configure(text='继续'); self.status.set('已暂停；点击继续沿用当前窗口和板端会话。')
        elif phase == 'running' and not self.cancelled.is_set(): self.status.set('运行中 · '+self.market_view.accuracy_text.get())
        if snapshot.get('error'): self.status.set('已停止：'+snapshot['error']+'；已保留日志，不会自动重试。')
        selected = self.market_view.selected
        for iid in self.history.get_children(): self.history.delete(iid)
        for row in reversed(snapshot['recent_predictions']):
            npu = row['outputs'][BACKEND]; outcome = row.get('outcome', {})
            actual = CLASS_CN[outcome['actual_class']] if outcome.get('status') == 'resolved' else '等待' if not outcome else '未兑现'
            correct = outcome.get('matches', {}).get(BACKEND)
            verdict = '命中' if correct is True else '未命中' if correct is False else '—'
            self.history.insert('', 'end', iid=str(row['window_id']), values=(row['window_id'],
                exchange_time(row['last']['timestamp_ns_since_exchange_midnight']), CLASS_CN[npu['class_index']],
                actual, verdict, f"{npu['elapsed_ns']/1e6:.3f}"))
        if selected and self.history.exists(str(selected)): self.history.selection_set(str(selected))
        if snapshot['recent_predictions']:
            values = snapshot['recent_predictions'][-1]['book']
            for i in range(10):
                ask, aq, bid, bq = values[i*4:i*4+4]
                self.book.item(str(i), values=(i+1, f'{ask/10000:.4f}', aq, f'{bid/10000:.4f}', bq))
        if self.market_view.dirty: self.market_view.draw()

    def select_prediction(self, event=None):
        selected = self.history.selection()
        if selected: self.market_view.select(int(selected[0])); self.market_view.draw()

    def poll(self):
        snapshot = self.mailbox.take()
        if snapshot: self.apply_snapshot(snapshot)
        if self.market_view.dirty: self.market_view.draw()
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'result':
                    self.run_directory = Path(value['directory'])
                    label = {'completed': '行情回放完成', 'partial': '已停止并保存', 'failed': '运行故障，结果已保存'}.get(value['status'], value['status'])
                    self.status.set(label+' · '+str(self.run_directory))
                    self.pause_button.configure(state='disabled'); self.stop_button.configure(state='disabled')
                elif kind == 'error':
                    self.status.set('运行已停止：'+value+'；本次会话不可重试。')
                    self.pause_button.configure(state='disabled'); self.stop_button.configure(state='disabled')
                elif kind == 'exported':
                    self.export_button.configure(state='normal'); self.status.set(f'已导出 {value[0]:,} 条预测：{value[1]}')
                elif kind == 'export_error':
                    self.export_button.configure(state='normal'); self.status.set('导出失败：'+value)
        except queue.Empty: pass
        if self.closing and not any(t and t.is_alive() for t in (self.thread, self.export_thread)):
            self.destroy(); return
        self.after(100, self.poll)

    def export_log(self):
        if not self.run_directory:
            messagebox.showinfo('导出预测', '开始回放并形成预测后可导出。', parent=self); return
        if self.export_thread and self.export_thread.is_alive(): return
        destination = filedialog.asksaveasfilename(parent=self, defaultextension='.csv', initialfile='FinEdge预测记录.csv', filetypes=[('CSV', '*.csv')])
        if not destination: return
        directory = self.run_directory
        limits = {name: (directory/(name+'.jsonl')).stat().st_size for name in ('predictions', 'outcomes')}
        self.export_button.configure(state='disabled')
        def work():
            try: self.messages.put(('exported', (export_showcase_csv(directory, destination, limits=limits), destination)))
            except Exception as exc: self.messages.put(('export_error', str(exc)))
        self.export_thread = threading.Thread(target=work, name='showcase-export', daemon=False); self.export_thread.start()

    def close_app(self):
        self.closing = True
        if self.thread and self.thread.is_alive(): self.stop()
        elif not (self.export_thread and self.export_thread.is_alive()): self.destroy()

    def show_help(self):
        messagebox.showinfo('怎么看', '① 左侧是已回放盘口的中间价K线，可切换为走势曲线。\n'
            '② 右侧是NPU判断：再过50条有效盘口后，价格跌、平还是涨；分数不是概率。\n'
            '③ 点击记录：A是预测时价格，B是未来第50条盘口的实际价格。未来尚未到达时不判对错。\n'
            '④ NPU时间包含启动、等待与输出读取；板内时间还包含FPGA组窗和记录DMA；批次往返另计网络，不可混为单窗NPU时间。\n'
            '⑤ 暂停/继续保留会话；停止后保存日志。新启动器自动软件重启板卡建立新会话。\n'
            +self.data_note.get()+'。\n'
            '12月15日原始行情按当前程序运行时重建，尚未与队友原验证缓存逐条对齐，不能复用其历史准确率。', parent=self)
