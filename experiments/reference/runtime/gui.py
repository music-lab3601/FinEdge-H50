"""Tk desktop demo; GUI polling stays independent of all inference work."""
import csv
from collections import deque
from contextlib import closing
import json
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from shared.model_paths import CLASS_CN
from software.runtime.delivery import write_json
from experiments.reference.runtime.replay import ReplayControl, run_replay
from experiments.reference.runtime.datasets import dataset_info
from experiments.reference.runtime.itch import DATASET
from experiments.reference.runtime.market_view import MarketView, npu_output
from experiments.reference.runtime.market import exchange_time
from experiments.reference.runtime.ui_widgets import BG, PANEL, CARD, TEXT, MUTED, ACCENT
from experiments.reference.runtime.demo_rate import validate_rate, MAX_TARGET_RATE
from experiments.reference.runtime.fpga_streaming import LatestDisplay

DATA_LABELS = {'itch20190730': 'AAPL · 2019-07-30 · 原始ITCH在线建簿',
               'itch20190730_gzip': 'AAPL · 2019-07-30 · 完整gzip在线建簿',
               'nasdaq20190730': 'AAPL · 2019-07-30 · 盘口文件参考',
               'delivery1024': 'AAPL · 2025-12-08 · 交付参考片段'}
MODE_LABELS = {'socket': 'CPU + 板载NPU（推荐）', 'cpu': 'CPU参考（不需板卡）',
               'socket-verify': 'CPU + Host + NPU校验', 'host': 'CPU + Host仿真', 'onnx': 'CPU + ONNX对照',
               'fpga': 'FPGA组窗 + 板载NPU · 50窗演示',
               'fpga-stream': 'FPGA组窗 + 板载NPU · 全日流式'}


def export_predictions_csv(directory, destination, *, limits=None):
    """Stream a fixed byte-prefix of the logs; safe to run off the Tk thread."""
    directory = Path(directory)
    paths = {name: directory/(name+'.jsonl') for name in ('predictions', 'outcomes')}
    limits = limits or {name: path.stat().st_size if path.is_file() else 0 for name, path in paths.items()}
    def rows(name):
        if not paths[name].is_file():
            return
        with paths[name].open('rb') as handle:
            remaining = limits[name]
            while remaining > 0:
                line = handle.readline(remaining)
                if not line:
                    break
                remaining -= len(line)
                # A concurrent writer's final partial line is outside the
                # captured complete prefix; malformed completed rows fail.
                if not line.endswith(b'\n'):
                    break
                yield json.loads(line)
    count = 0
    with closing(rows('outcomes')) as outcomes, closing(rows('predictions')) as predictions, open(destination, 'w', encoding='utf-8-sig', newline='') as dst:
        next_outcome = next(outcomes, None)
        writer = csv.writer(dst)
        writer.writerow(('run_id', 'window_id', 'round', 'source_row', 'cpu_class', 'host_class', 'npu_class',
                         'conversion_max_abs_error', 'npu_max_abs_error', 'cpu_npu_max_abs_error', 'host_check_performed',
                         'date', 'origin_exchange_ns', 'origin_midpoint_usd', 'outcome_status', 'actual_class',
                         'actual_source_row', 'actual_exchange_ns', 'actual_midpoint_usd', 'cpu_correct', 'npu_correct', 'unresolved_reason'))
        label = lambda result: result.get('class_name', CLASS_CN[result['class_index']]) if result else ''
        for row in predictions:
            outputs = row['outputs']
            while next_outcome is not None and next_outcome['window_id'] < row['window_id']:
                next_outcome = next(outcomes, None)
            outcome = next_outcome if next_outcome is not None and next_outcome['window_id'] == row['window_id'] else {}
            actual, matches = outcome.get('actual_meta', {}), outcome.get('matches', {})
            writer.writerow((row.get('run_id', ''), row['window_id'], row.get('round', 1), row['last']['source_record_row_0based'],
                             label(outputs.get('torch_cpu')), label(outputs.get('icraft_host')), label(npu_output(outputs)),
                             row.get('conversion_max_abs_error', '') if row.get('conversion_comparison_available', False) else '',
                             row.get('npu_max_abs_error', ''), row.get('cpu_npu_max_abs_error', ''),
                             row.get('npu_within_tolerance') is not None,
                             row['last']['date'], row['last']['timestamp_ns_since_exchange_midnight'],
                             row['origin_mid2']/20000 if 'origin_mid2' in row else '', outcome.get('status', 'pending'),
                             CLASS_CN[outcome['actual_class']] if 'actual_class' in outcome else '',
                             actual.get('source_record_row_0based', ''), actual.get('timestamp_ns_since_exchange_midnight', ''),
                             outcome['actual_mid2']/20000 if 'actual_mid2' in outcome else '', matches.get('torch_cpu', ''),
                             matches.get('board_socket_npu', matches.get('icraft_socket', '')), outcome.get('reason', '')))
            count += 1
    return count


class App(tk.Tk):
    def __init__(self, fpga_session=None, fpga_stream_session=None, *, stream_runner=None):
        super().__init__()
        self.fpga_session = fpga_session
        self.fpga_stream_session = fpga_stream_session
        self.stream_runner = stream_runner
        self.stream_display = LatestDisplay()
        self.stream_consumed = False
        self.stream_snapshot = None
        self.export_thread = None
        self.fpga_consumed = False
        self.title('FinEdge 0.7 · 原始事件在线建簿与 FPGA / NPU 推理')
        self.geometry('1280x900')
        self.minsize(1050, 760)
        self.configure(bg=BG)
        self.messages = queue.Queue()
        self.thread = None
        self.control = None
        self.run_directory = None
        self.ui_report_saved = False
        self.last_event = None
        self.displayed = 0
        self.history_ids = deque()
        self.heartbeat_count = 0
        self.last_poll_ns = time.perf_counter_ns()
        self.ui_metrics = {'polls': 0, 'max_heartbeat_gap_ms': 0.0, 'max_render_batch_ms': 0.0,
                           'max_queue_depth': 0}
        self.mode = tk.StringVar(value='socket')
        default_dataset = ('itch20190730' if (DATASET/'extracted/manifest.json').is_file() else
                           'nasdaq20190730' if (DATASET/'prepared/manifest.json').is_file() else 'delivery1024')
        self.dataset = tk.StringVar(value=default_dataset)
        self.rate = tk.DoubleVar(value=10)
        self.loop = tk.BooleanVar(value=True)
        self.state_text = tk.StringVar(value='就绪')
        self.dataset_label = tk.StringVar(value=DATA_LABELS[self.dataset.get()])
        self.mode_label = tk.StringVar(value=MODE_LABELS[self.mode.get()])
        self.progress_text = tk.StringVar(value='窗口 0 · 已处理盘口 0 · 第0轮')
        self.history_filter = tk.StringVar(value='已核对')
        self.status = tk.StringVar(value='就绪 · 选择模式后开始；socket 模式需要板卡在线')
        self.counter = tk.StringVar(value='累计记录 0  ·  窗口 0  ·  轮次 0')
        self.prediction = tk.StringVar(value='CPU —    Host —    NPU —')
        self.detail = tk.StringVar(value='模型：ED4 seed42 / epoch3    输入：40 × 100    类别：跌 / 平 / 涨')
        self.timing = tk.StringVar(value='CPU — ms    Host — ms    socket — ms')
        self.error_text = tk.StringVar(value='CPU/NPU一致性与预测是否正确分开统计；未来50条到齐才判定真实方向。')
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=TEXT, font=('Microsoft YaHei UI', 9))
        style.configure('Section.TLabel', font=('Microsoft YaHei UI', 11, 'bold'))
        style.configure('Price.TLabel', foreground=TEXT, font=('Consolas', 22, 'bold'))
        style.configure('TButton', font=('Microsoft YaHei UI', 9), padding=(9, 5), background=CARD, foreground=TEXT, borderwidth=0)
        style.map('TButton', background=[('active', '#29476b')], foreground=[('disabled', MUTED)])
        style.configure('Primary.TButton', background=ACCENT, foreground=BG)
        style.map('Primary.TButton', background=[('disabled', '#2a3d57'), ('active', '#97c3ff')])
        style.configure('TCombobox', fieldbackground=PANEL, background=CARD, foreground=TEXT, padding=4)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', TEXT)])
        style.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=25,
                        borderwidth=0, font=('Microsoft YaHei UI', 9))
        style.configure('Treeview.Heading', background=CARD, foreground=MUTED, font=('Microsoft YaHei UI', 9, 'bold'), padding=5)
        style.map('Treeview', background=[('selected', '#2b4e77')], foreground=[('selected', '#ffffff')])
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=MUTED, padding=(12, 5))
        style.map('TNotebook.Tab', background=[('selected', CARD)], foreground=[('selected', TEXT)])
        self._build()
        if fpga_session:
            self.mode.set('fpga')
            self.mode_label.set(MODE_LABELS['fpga'])
            self.choose_mode()
        if fpga_stream_session:
            self.mode.set('fpga-stream')
            self.mode_label.set(MODE_LABELS['fpga-stream'])
            self.choose_mode()
        self.bind('<Configure>', self.resize_dashboard)
        self.protocol('WM_DELETE_WINDOW', self.close_app)
        self.after(50, self.poll)

    def _build(self):
        container = ttk.Frame(self, padding=(16, 10))
        container.pack(fill='both', expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)
        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky='ew', pady=(0, 7))
        tk.Label(header, text='FinEdge', bg=BG, fg=TEXT, font=('Segoe UI', 23, 'bold')).pack(side='left')
        ttk.Label(header, text='订单流预测观察台', foreground=MUTED).pack(side='left', padx=12)
        self.state_badge = tk.Label(header, textvariable=self.state_text, bg='#20364f', fg=ACCENT,
                                   padx=12, pady=4, font=('Microsoft YaHei UI', 9, 'bold'))
        self.state_badge.pack(side='right')
        ttk.Button(header, text='怎么看这个界面？', command=self.show_help).pack(side='right', padx=10)
        config = ttk.Frame(container)
        config.grid(row=1, column=0, sticky='ew')
        ttk.Label(config, text='行情').pack(side='left')
        self.dataset_box = ttk.Combobox(config, textvariable=self.dataset_label,
                                        values=tuple(DATA_LABELS.values()), state='readonly', width=33)
        self.dataset_box.pack(side='left', padx=(6, 12))
        self.dataset_box.bind('<<ComboboxSelected>>', self.choose_dataset)
        ttk.Label(config, text='推理').pack(side='left')
        self.mode_box = ttk.Combobox(config, textvariable=self.mode_label,
                                    values=tuple(MODE_LABELS.values()), state='readonly', width=24)
        self.mode_box.pack(side='left', padx=(6, 12))
        self.mode_box.bind('<<ComboboxSelected>>', self.choose_mode)
        ttk.Label(config, text='目标窗/秒').pack(side='left')
        ttk.Spinbox(config, from_=0, to=MAX_TARGET_RATE, textvariable=self.rate, width=5, command=self.set_rate).pack(side='left', padx=6)
        ttk.Label(config, text='0=不限速', foreground=MUTED).pack(side='left', padx=(0,4))
        tk.Checkbutton(config, text='循环回放', variable=self.loop, bg=BG, fg=MUTED, selectcolor=PANEL,
                       activebackground=BG, activeforeground=TEXT, font=('Microsoft YaHei UI', 9)).pack(side='left')
        control_row = ttk.Frame(container)
        control_row.grid(row=2, column=0, sticky='ew', pady=(7, 6))
        controls = ttk.Frame(control_row)
        controls.pack(fill='x')
        self.start_button = ttk.Button(controls, text='开始回放', command=self.start, style='Primary.TButton')
        self.start_button.pack(side='left', padx=(0, 5))
        self.pause_button = ttk.Button(controls, text='暂停', command=self.pause)
        self.pause_button.pack(side='left', padx=5)
        for label, command in (('重置', self.reset_replay), ('停止', self.stop), ('导出CSV', self.export_log)):
            ttk.Button(controls, text=label, command=command).pack(side='left', padx=5)
        ttk.Label(controls, text='记录').pack(side='left', padx=(12, 5))
        filter_box = ttk.Combobox(controls, textvariable=self.history_filter,
                                  values=('已核对', '全部（含等待）'), state='readonly', width=13)
        filter_box.pack(side='left')
        filter_box.bind('<<ComboboxSelected>>', self.filter_history)
        ttk.Label(control_row, textvariable=self.progress_text, foreground=MUTED).pack(anchor='w', pady=(3, 0))
        self.market_view = MarketView(container)
        self.market_view.grid(row=3, column=0, sticky='nsew')
        self.chart = self.market_view.canvas
        self.tabs = ttk.Notebook(container, height=122)
        self.tabs.grid(row=4, column=0, sticky='ew', pady=(8, 4))
        log_frame, book_frame = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        technical = ttk.Frame(self.tabs, padding=7)
        self.help_frame = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(log_frame, text='预测记录 · 点一行复盘')
        self.tabs.add(book_frame, text='十档盘口')
        self.tabs.add(technical, text='工程详情')
        self.tabs.add(self.help_frame, text='怎么看')
        self.history = ttk.Treeview(log_frame, columns=('id', 'round', 'time', 'cpu', 'npu', 'actual', 'match', 'error'),
                                    displaycolumns=('id', 'time', 'cpu', 'npu', 'actual', 'match'),
                                    show='headings', height=4, selectmode='browse')
        for c, title, width in zip(self.history['columns'],
                ('窗口', '轮次', '发出预测的时刻（纽约）', 'CPU预测', 'NPU / 对照预测', '50条后实际', 'CPU / NPU 判定', 'NPU/Host误差'),
                (70, 60, 180, 100, 120, 120, 180, 110)):
            self.history.heading(c, text=title)
            self.history.column(c, width=width, anchor='center')
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.history.yview)
        self.history.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.history.pack(fill='both', expand=True)
        self.history.tag_configure('correct', foreground='#85dfc7')
        self.history.tag_configure('incorrect', foreground='#f4c78e')
        self.history.tag_configure('pending', foreground=MUTED)
        self.history.bind('<<TreeviewSelect>>', self.select_prediction)
        self.market_view.on_follow = lambda: self.history.selection_remove(*self.history.selection())
        self.book = ttk.Treeview(book_frame, columns=('level', 'ask', 'askqty', 'bid', 'bidqty'), show='headings', height=4)
        for c, title in zip(self.book['columns'], ('档位', '卖价 USD', '卖量（股）', '买价 USD', '买量（股）')):
            self.book.heading(c, text=title)
            self.book.column(c, width=150 if c != 'level' else 60, anchor='e')
        for i in range(10):
            self.book.insert('', 'end', iid=str(i), values=(i+1, '—', '—', '—', '—'))
        book_scroll = ttk.Scrollbar(book_frame, orient='vertical', command=self.book.yview)
        self.book.configure(yscrollcommand=book_scroll.set)
        book_scroll.pack(side='right', fill='y')
        self.book.pack(fill='both', expand=True)
        for var in (self.prediction, self.detail, self.timing, self.error_text, self.counter, self.status):
            ttk.Label(technical, textvariable=var, wraplength=1000, font=('Microsoft YaHei UI', 8)).pack(anchor='w')
        help_text = (
            '① 看左侧真实行情：蓝线是已发生的盘口中间价；想看蜡烛图，把“中间价走势”切换为“K线”。\n'
            '② 看右侧最新预测：CPU和NPU是在判断再过50条盘口后涨、平还是跌。“两路一致”不等于“预测正确”。\n'
            '③ 点预测记录中的一行：下面三张复盘卡依次是当时价格、当时预测、50条后的实际结果。A到B是实际变化。\n'
            '④ 等待中的窗口先不判断对错；命中率只统计已有实际结果的窗口。H50是50条记录，不是50秒或下一根K线。\n'
            '建议：先运行20秒 → 暂停 → 选择一条已有实际结果的记录。行情时间与电脑运行时间不同，100毫秒周期更易观察。')
        ttk.Label(self.help_frame, text=help_text, justify='left', wraplength=1000, font=('Microsoft YaHei UI', 9)).pack(anchor='w')
        self.source_text = tk.StringVar(value=dataset_info(self.dataset.get())['description'])
        ttk.Label(container, text='历史有效盘口中间价（非成交价） · 当前样本命中率不代表独立样本外准确率或交易收益',
                  foreground=MUTED, font=('Microsoft YaHei UI', 8)).grid(row=5, column=0, sticky='w')

    def resize_dashboard(self, event):
        if event.widget is self:
            compact = event.height < 870
            if getattr(self, '_compact', None) != compact:
                self._compact = compact
                self.market_view.set_compact(compact)

    def choose_dataset(self, _event=None):
        self.dataset.set(next(key for key, value in DATA_LABELS.items() if value == self.dataset_label.get()))

    def choose_mode(self, _event=None):
        self.mode.set(next(key for key, value in MODE_LABELS.items() if value == self.mode_label.get()))
        if self.mode.get() in ('fpga', 'fpga-stream'):
            self.dataset.set('itch20190730')
            self.dataset_label.set(DATA_LABELS['itch20190730'])
            self.loop.set(False)
            self.rate.set(20)
            self.history_filter.set('全部（含等待）')
            self.filter_history()
            self.status.set('有限硬件演示仍为50窗；默认20窗/秒，可设0不限速。全日模式待缓冲池验收。')
            if self.mode.get() == 'fpga-stream':
                self.rate.set(0)
                self.status.set('全日流式 · 0不限速；暂停/停止在当前批次完成后生效；会话不重置、不自动重连。')
        consumed = (self.mode.get() == 'fpga' and self.fpga_consumed or
                    self.mode.get() == 'fpga-stream' and self.stream_consumed)
        self.start_button.configure(state='disabled' if consumed else 'normal')

    def show_help(self):
        self.tabs.select(self.help_frame)

    def filter_history(self, _event=None):
        for row in self.history_ids:
            if not self.history.exists(row):
                continue
            ready = self.history.item(row, 'values')[5] in CLASS_CN
            if self.history_filter.get() == '已核对' and not ready:
                self.history.detach(row)
            else:
                self.history.move(row, '', 0)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        if self.mode.get() == 'fpga':
            if self.fpga_consumed or not self.fpga_session:
                self.status.set('请使用“启动FPGA闭环演示.cmd”。硬件会话一次性使用，不能重新开始或重置。')
                return
            self.dataset.set('itch20190730')
            self.loop.set(False)
            self.history_filter.set('全部（含等待）')
        if self.mode.get() == 'fpga-stream':
            if self.stream_consumed or not self.fpga_stream_session:
                self.status.set('请使用已准备的全日流式会话启动；本窗口不重置、重连或再次使用硬件会话。')
                return
            self.dataset.set('itch20190730')
            self.loop.set(False)
            self.history_filter.set('全部（含等待）')
            self.stream_display = LatestDisplay()
            self.stream_snapshot = None
        try:
            rate = validate_rate(self.rate.get())
        except (ValueError, tk.TclError):
            self.status.set('目标速度请输入0～1000窗/秒；0为不限速，实际速度取决于处理能力。')
            return
        self.market_view.clear(new_run=True)
        self.state_text.set('准备中')
        self.dataset_label.set(DATA_LABELS[self.dataset.get()])
        self.mode_label.set(MODE_LABELS[self.mode.get()])
        self.progress_text.set('正在校验数据、加载模型并预热…')
        self.displayed = 0
        self.run_directory = None
        self.ui_report_saved = False
        self.last_poll_ns = time.perf_counter_ns()
        self.ui_metrics = {'polls': 0, 'max_heartbeat_gap_ms': 0.0, 'max_render_batch_ms': 0.0,
                           'max_queue_depth': 0}
        self.pause_button.configure(text='暂停')
        for row in self.history_ids:
            if self.history.exists(row):
                self.history.delete(row)
        self.history_ids.clear()
        self.control = ReplayControl(rate)
        self.status.set('正在校验输入、从完整事件前缀建簿并准备首窗；随后加载后端及预热…')
        self.start_button.configure(state='disabled')
        self.mode_box.configure(state='disabled')
        self.dataset_box.configure(state='disabled')
        mode, loop, dataset = self.mode.get(), self.loop.get(), self.dataset.get()
        if mode == 'fpga':
            self.fpga_consumed = True
        if mode == 'fpga-stream':
            self.stream_consumed = True
        self.source_text.set(dataset_info(dataset)['description'])
        def work():
            try:
                if mode == 'fpga':
                    from experiments.reference.runtime.fpga_demo_replay import run_fpga_demo
                    run_fpga_demo(self.fpga_session, control=self.control, emit=self.messages.put)
                elif mode == 'fpga-stream':
                    runner = self.stream_runner
                    if runner is None:
                        from experiments.reference.runtime.ps_stream_replay import run_ps_stream
                        runner = run_ps_stream
                    def lifecycle(item):
                        if item['type'] in ('ready', 'error', 'finished'):
                            self.messages.put({'type': 'stream_lifecycle', 'event': item})
                    runner(session_path=self.fpga_stream_session, on_event=lifecycle,
                           display=self.stream_display, paused=self.control.pause,
                           cancelled=self.control.stop, get_rate=lambda: self.control.rate,
                           max_windows=10000)
                else:
                    run_replay(mode=mode, rounds=1, control=self.control, emit=self.messages.put,
                               dataset=dataset,repeat=loop)
            except Exception as exc:
                # run_replay has already logged and queued the complete failure.
                if mode in ('fpga', 'fpga-stream'):
                    self.messages.put({'type':'demo_init_failed','message':str(exc)})
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def pause(self):
        if not self.control:
            return
        if self.control.pause.is_set():
            self.control.pause.clear()
            self.pause_button.configure(text='暂停')
            self.status.set('运行中')
            self.state_text.set('运行中')
        else:
            self.control.pause.set()
            self.pause_button.configure(text='继续')
            self.status.set('已暂停 · 窗口和推理会话保留')
            self.state_text.set('已暂停 · 可选窗复盘')
            if self.mode.get() == 'fpga-stream':
                self.status.set('已请求暂停 · 当前批次确认后暂停；不会中断在途 FPGA / NPU 调用')
                self.state_text.set('等待批次边界暂停')

    def reset_replay(self):
        if self.mode.get() in ('fpga', 'fpga-stream'):
            self.status.set('此硬件会话不支持重置或自动重连。可暂停/继续，结束后查看或导出结果。')
            return
        if self.control and self.thread and self.thread.is_alive():
            self.control.reset.set()
            self.control.pause.clear()
            self.pause_button.configure(text='暂停')
            self.status.set('重置到片段开头，重新积累100条窗口；历史日志保留')
            self.state_text.set('重新建簿/积累窗口中')

    def set_rate(self):
        if self.control:
            try:
                rate = validate_rate(self.rate.get())
                self.control.rate = rate
            except (ValueError, tk.TclError):
                pass

    def stop(self):
        if self.control:
            self.control.stop.set()
            self.control.pause.clear()
            self.status.set('正在停止，等待当前推理返回…')
            if self.mode.get() == 'fpga-stream':
                self.status.set('正在停止：等待当前批次确认并保存结果；完整日未完成时明确记录 stopped。')

    def export_log(self):
        if not self.run_directory:
            messagebox.showinfo('日志', '请先运行一次回放。')
            return
        if self.export_thread and self.export_thread.is_alive():
            self.status.set('CSV仍在后台导出；回放和控制可继续使用。')
            return
        destination = filedialog.asksaveasfilename(defaultextension='.csv', initialfile='finedge_predictions.csv', filetypes=[('CSV', '*.csv')])
        if destination:
            self.start_export(destination)

    def start_export(self, destination):
        if self.export_thread and self.export_thread.is_alive():
            return
        directory = self.run_directory
        limits = {name: (directory/(name+'.jsonl')).stat().st_size
                  if (directory/(name+'.jsonl')).is_file() else 0
                  for name in ('predictions', 'outcomes')}
        self.status.set('后台导出已写入的完整日志前缀；新产生的数据留待下一次导出。')
        def work():
            try:
                count = export_predictions_csv(directory, destination, limits=limits)
                self.messages.put({'type': 'export_finished', 'destination': str(destination), 'rows': count})
            except Exception as exc:
                self.messages.put({'type': 'export_failed', 'message': str(exc)})
        self.export_thread = threading.Thread(target=work, name='csv-log-export', daemon=True)
        self.export_thread.start()

    def consume(self, item):
        kind = item['type']
        if kind == 'export_finished':
            self.status.set(f"已后台导出 {item['rows']:,} 个窗口：{item['destination']}")
            return
        if kind == 'export_failed':
            self.status.set('CSV导出失败：' + item['message'])
            return
        if kind == 'stream_lifecycle':
            event = item['event']
            if event.get('directory'):
                self.run_directory = Path(event['directory'])
            if event['type'] == 'error':
                self.state_text.set('流式执行故障')
                self.status.set(str(event.get('message', '执行故障'))[:500])
            elif event['type'] == 'finished':
                self.start_button.configure(state='disabled')
                self.mode_box.configure(state='disabled')
                self.pause_button.configure(text='暂停')
            return
        if kind == 'demo_init_failed':
            self.state_text.set('启动失败')
            self.status.set(item['message'])
        if kind == 'demo_phase':
            self.state_text.set('硬件推理完成 · 兑现H50')
            self.status.set(item['message'])
        if kind in ('market', 'round', 'prediction', 'outcome', 'finished'):
            self.market_view.consume(item)
        if kind == 'source_progress':
            source = item['source']
            self.status.set(f"原始流已处理 {source['selected_events']:,} 个AAPL/控制事件 · "
                            f"建簿 {source['mutation_rows']:,} 行 · 有效盘口 {source['eligible_records']:,} · 文件背压验证")
            if self.displayed == 0:
                self.progress_text.set(f"从事件前缀建簿中 · 已解析 {source['frames_read']:,} 个事件")
        elif kind == 'ready':
            self.state_text.set('运行中')
            self.run_directory = Path(item['directory'])
            self.status.set('运行中 · ' + ', '.join(item['backends']))
            if 'dataset' in item:
                count = item['source_count']
                self.source_text.set(item['dataset']['description']+
                                     (f" · 有效记录{count:,}条" if count is not None else ' · 有效记录数随事件处理发现'))
        elif kind == 'round':
            self.state_text.set('建簿/积累窗口中')
            self.status.set(f"第 {item['round']} 轮 · 从头积累窗口；不是新交易日数据")
        elif kind == 'warming_window':
            self.counter.set(f"积累窗口 {item['count']} / 100 · 第 {item['round']} 轮")
            self.progress_text.set(f"积累窗口 {item['count']} / 100 · 第{item['round']}轮")
        elif kind == 'prediction':
            if self.control is not None and not self.control.pause.is_set():
                self.state_text.set('运行中')
            self.last_event = item
            self.displayed += 1
            stats, outputs = item['stats'], item['outputs']
            self.progress_text.set(f"窗口 {stats['windows']:,} · 盘口 {stats['records']:,} · 第{item['round']}轮")
            cpu = outputs['torch_cpu']
            host = outputs.get('icraft_host', outputs.get('onnx_cpu'))
            npu = outputs.get('icraft_socket')
            label = lambda p: CLASS_CN[p['class_index']] if p else '—'
            self.prediction.set(f"CPU {label(cpu)}     Host {label(host) if host else '未运行'}     NPU {label(npu)}")
            self.counter.set(f"累计记录 {stats['records']} · 累计窗口 {stats['windows']} · 第 {item['round']} 轮 · 源盘口行 {item['last']['source_record_row_0based']} · 原始消息位置 {item['last']['source_message_position_1based']}")
            self.detail.set('CPU logits: ' + ' / '.join(f'{v:+.5f}' for v in cpu['logits']) +
                            ('    NPU: ' + ' / '.join(f'{v:+.5f}' for v in npu['logits']) if npu else ''))
            ms = lambda p: f"{p['elapsed_ns']/1e6:.3f}" if p else '—'
            self.timing.set(f"CPU {ms(cpu)} ms    Host {ms(host)} ms    socket {ms(npu)} ms    预处理 {item['preprocess_ns']/1e6:.3f} ms")
            err = item['npu_max_abs_error']
            err_text = f'{err:.6g}' if err is not None else ('未校验' if npu else '未启用')
            if npu and not host:
                self.error_text.set(f"CPU/NPU分数差 {item['cpu_npu_max_abs_error']:.6g} · 类别差异累计 {stats['cpu_npu_class_mismatches']} · Host未校验（socket-verify执行三路验收）")
                if item.get('live_hardware'):
                    self.state_text.set('FPGA → NPU · 真实运行')
                    self.error_text.set(f"FPGA张量及身份已核对 · CPU/NPU分数差 {item['cpu_npu_max_abs_error']:.6g} · 类别差异 {stats['cpu_npu_class_mismatches']} · Host不逐窗运行")
                    if item.get('compiled_reference_checked'):
                        self.error_text.set(f"FPGA张量及身份已核对 · NPU/已保存Host误差 {item['npu_max_abs_error']:.6g} · CPU/NPU分数差 {item['cpu_npu_max_abs_error']:.6g} · Host仅查参考")
                    self.timing.set(f"板卡往返 {ms(npu)} ms · 含FPGA组窗/板内适配/NPU · CPU独立参考 {ms(cpu)} ms")
                    observed = item.get('observed_windows_per_second')
                    actual = f'{observed:.1f}' if observed is not None else '统计中'
                    target = item.get('target_windows_per_second', 0)
                    target_text = '不限速' if target == 0 else f'{target:g}窗/秒'
                    self.progress_text.set(f"窗口 {stats['windows']} · 目标{target_text} · 实际 {actual}窗/秒")
            elif host:
                self.error_text.set(f"转换误差 {item['conversion_max_abs_error']:.6g} · NPU/Host误差 {err_text} · NPU超差 {stats['npu_mismatches']} · CPU/转换模型类别差异 {stats['conversion_class_mismatches']}")
            else:
                self.error_text.set('CPU参考模式 · 未运行转换模型及NPU；真实方向和预测命中情况见K线下方。')
            book = item['book']
            for i in range(10):
                ask, ask_qty, bid, bid_qty = book[4*i:4*i+4]
                self.book.item(str(i), values=(i+1, f'{ask/10000:.4f}', ask_qty, f'{bid/10000:.4f}', bid_qty))
            self.history.insert('', 0, iid=str(item['window_id']), values=(item['window_id'], item['round'],
                exchange_time(item['last']['timestamp_ns_since_exchange_midnight']), label(cpu), label(npu or host), '等待50条', '等待结果', err_text), tags=('pending',))
            self.history_ids.append(str(item['window_id']))
            if self.history_filter.get() == '已核对':
                self.history.detach(str(item['window_id']))
            while len(self.history_ids) > 150:
                self.history.delete(self.history_ids.popleft())
        elif kind == 'outcome':
            row = str(item['window_id'])
            if self.history.exists(row):
                values = list(self.history.item(row, 'values'))
                values[5] = CLASS_CN[item['actual_class']] if item['status'] == 'resolved' else '未兑现'
                match = lambda name: ('命中' if item['matches'][name] else '未命中') if name in item.get('matches', {}) else '未启用'
                values[6] = match('torch_cpu') + ' / ' + match('icraft_socket')
                tag = 'correct' if item.get('matches') and all(item['matches'].values()) else 'incorrect' if item.get('matches') else 'pending'
                self.history.item(row, values=values, tags=(tag,))
                if self.history_filter.get() == '已核对' and item['status'] == 'resolved':
                    self.history.move(row, '', 0)
        elif kind == 'error':
            self.state_text.set('连接或执行故障')
            self.status.set('故障：' + item['message'][:240])
            self.run_directory = Path(item['directory'])
        elif kind == 'finished':
            self.run_directory = Path(item['directory'])
            summary = item['summary']
            if self.mode.get() == 'fpga':
                self.progress_text.set(f"硬件窗口 {summary['stats']['windows']} · 观察盘口 {summary['stats']['records']} · 本次结束")
            self.state_text.set('运行故障' if summary['status'] == 'failed' else
                                '证据未收全' if summary['status'] == 'incomplete_evidence' else '回放已结束')
            self.status.set(f"{summary['status']} · 已保存 {summary['stats']['windows']} 个窗口：{self.run_directory.name}" +
                            (' · ' + summary.get('error', '')[:160] if summary['status'] == 'failed' else ''))
            self.start_button.configure(state='disabled' if self.mode.get() == 'fpga' else 'normal')
            self.mode_box.configure(state='disabled' if self.mode.get() == 'fpga' else 'readonly')
            self.dataset_box.configure(state='readonly')
            self.pause_button.configure(text='暂停')

    def draw_chart(self):
        self.market_view.draw()

    def consume_stream_snapshot(self, snapshot):
        if snapshot.get('schema') != 1:
            raise ValueError('Unknown PS stream display schema')
        self.stream_snapshot = snapshot
        self.market_view.apply_snapshot(snapshot)
        if snapshot.get('directory'):
            self.run_directory = Path(snapshot['directory'])
        stats = snapshot['stats']
        windows, records = stats.get('windows', 0), stats.get('records', 0)
        self.displayed = windows
        self.ui_metrics['stream_snapshots_displayed'] = self.ui_metrics.get('stream_snapshots_displayed', 0)+1
        self.ui_metrics['stream_snapshots_coalesced'] = self.stream_display.coalesced
        rate = snapshot.get('target_windows_per_second', 0)
        actual = snapshot.get('observed_windows_per_second')
        target = '不限速' if rate == 0 else f'{rate:g}窗/s'
        achieved = f'{actual:.1f}' if actual is not None else '统计中'
        self.progress_text.set(f"{windows:,} / {snapshot['expected_windows']:,} 窗 · 目标{target} · 实际{achieved}窗/s")
        phase = snapshot.get('phase', 'preparing')
        status = snapshot.get('status', 'initializing')
        names = {'preparing': '准备会话', 'building_book': '从原始事件建簿', 'running': 'FPGA → NPU · 全日运行',
                 'paused': '已在批次边界暂停', 'stopping': '正在保存并停止', 'failed': '流式执行故障',
                 'passed': '本次验证完成', 'passed_full_day': '完整交易日已完成', 'stopped': '已停止 · 非完整日',
                 'incomplete': '证据未收全', 'completed': '完整交易日已完成', 'partial': '已停止 · 非完整日',
                 'awaiting_continuation': '检查点已完成 · 等待继续'}
        if status == 'completed' and not snapshot.get('full_day_complete'):
            names['completed'] = '指定数据已完成 · 非全日实机验收'
        if self.control and self.control.pause.is_set() and phase == 'running':
            self.state_text.set('等待批次边界暂停')
        else:
            self.state_text.set(names.get(phase, phase))
        if phase in ('paused', 'awaiting_continuation'):
            self.pause_button.configure(text='继续')
        self.counter.set(f"累计盘口 {records:,} · 累计推理 {windows:,} · 显示缓存 {len(snapshot['recent_predictions'])} 行 · 合并显示帧 {self.stream_display.coalesced:,}（不丢推理）")
        memory = snapshot.get('memory', {})
        rss = lambda key: f"{memory[key]/1024:.1f} MiB" if isinstance(memory.get(key), (int, float)) else '未取得'
        detail = f"电脑 RSS {rss('host_rss_kib')} · 板端 RSS {rss('board_rss_kib')} · 错误 {stats.get('errors', 0)} · 单批在途 / 固定槽"
        if snapshot.get('error'):
            detail += ' · ' + str(snapshot['error'])[:350]
        self.status.set(detail)
        self.error_text.set(f"CPU/NPU 类别差异 {stats.get('cpu_npu_class_mismatches', 0):,} · 最大分数差 {stats.get('cpu_npu_max_abs_error', 0):.6g} · CPU逐窗参考；Host不逐窗执行")
        rows = snapshot['recent_predictions']
        if rows:
            latest = rows[-1]
            self.last_event = latest
            cpu, npu = latest['outputs'].get('torch_cpu'), npu_output(latest['outputs'])
            label = lambda p: CLASS_CN[p['class_index']] if p else '未执行'
            self.prediction.set(f"CPU {label(cpu)}     NPU {label(npu)}")
            scores = lambda p: ' / '.join(f'{v:+.5f}' for v in p['logits']) if p else '—'
            self.detail.set(f"CPU logits: {scores(cpu)}    NPU logits: {scores(npu)}")
            ms = lambda p: f"{p.get('elapsed_ns', 0)/1e6:.3f}" if p else '—'
            self.timing.set(f"CPU独立参考 {ms(cpu)} ms · NPU调用 {ms(npu)} ms · 实际吞吐含传输/组窗/日志；批均摊不等于单窗端到端")
            book = latest.get('book')
            if book:
                for i in range(10):
                    ask, ask_qty, bid, bid_qty = book[4*i:4*i+4]
                    self.book.item(str(i), values=(i+1, f'{ask/10000:.4f}', ask_qty, f'{bid/10000:.4f}', bid_qty))
        desired = {str(row['window_id']) for row in rows}
        for old in list(self.history_ids):
            if old not in desired and self.history.exists(old):
                self.history.delete(old)
        self.history_ids = deque(str(row['window_id']) for row in rows)
        for row in rows:
            identity = str(row['window_id'])
            outputs, outcome = row['outputs'], row.get('outcome', {})
            cpu, npu = outputs.get('torch_cpu'), npu_output(outputs)
            label = lambda p: CLASS_CN[p['class_index']] if p else '—'
            actual_label = CLASS_CN[outcome['actual_class']] if outcome.get('status') == 'resolved' else '未兑现' if outcome else '等待50条'
            matches = outcome.get('matches', {})
            verdict = lambda key: '命中' if matches.get(key) is True else '未命中' if matches.get(key) is False else '等待结果'
            verdict_text = verdict('torch_cpu')+' / '+verdict('board_socket_npu')
            error = row.get('cpu_npu_max_abs_error')
            values = (identity, row.get('round', 1), exchange_time(row['last']['timestamp_ns_since_exchange_midnight']),
                      label(cpu), label(npu), actual_label, verdict_text, f'{error:.6g}' if error is not None else '未校验')
            tag = 'correct' if matches and all(matches.values()) else 'incorrect' if matches else 'pending'
            if self.history.exists(identity):
                self.history.item(identity, values=values, tags=(tag,))
            else:
                self.history.insert('', 0, iid=identity, values=values, tags=(tag,))
            if self.history_filter.get() == '已核对' and outcome.get('status') != 'resolved':
                self.history.detach(identity)
            else:
                self.history.move(identity, '', 0)
        if status not in ('initializing', 'running'):
            self.start_button.configure(state='disabled')
            self.mode_box.configure(state='disabled')
            self.pause_button.configure(text='暂停')

    def select_prediction(self, _event=None):
        selected = self.history.selection()
        if selected:
            self.market_view.select(int(selected[0]))

    def poll(self):
        now = time.perf_counter_ns()
        self.ui_metrics['max_heartbeat_gap_ms'] = max(self.ui_metrics['max_heartbeat_gap_ms'],
                                                     (now-self.last_poll_ns)/1e6)
        self.last_poll_ns = now
        self.ui_metrics['polls'] += 1
        self.ui_metrics['max_queue_depth'] = max(self.ui_metrics['max_queue_depth'], self.messages.qsize())
        self.heartbeat_count += 1
        # Bound work per tick so high-throughput inference cannot starve controls.
        for _ in range(100):
            try:
                self.consume(self.messages.get_nowait())
            except queue.Empty:
                break
        if self.mode.get() == 'fpga-stream':
            snapshot = self.stream_display.take()
            if snapshot is not None:
                self.consume_stream_snapshot(snapshot)
        self.set_rate()
        if self.market_view.dirty:
            self.draw_chart()
        self.ui_metrics['max_render_batch_ms'] = max(self.ui_metrics['max_render_batch_ms'],
                                                   (time.perf_counter_ns()-now)/1e6)
        if self.run_directory and self.thread and not self.thread.is_alive() and self.messages.empty() and not self.ui_report_saved:
            write_json(self.run_directory / 'ui_metrics.json', {**self.ui_metrics,
                       'displayed_windows': self.displayed})
            self.ui_report_saved = True
        self.after(50, self.poll)

    def close_app(self):
        self.stop()
        def wait():
            if (self.thread and self.thread.is_alive()) or (self.export_thread and self.export_thread.is_alive()):
                self.after(100, wait)
            else:
                self.destroy()
        wait()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpga-session', type=Path)
    parser.add_argument('--fpga-stream-session', type=Path)
    args = parser.parse_args()
    if args.fpga_session and args.fpga_stream_session:
        parser.error('Select one hardware session type')
    App(args.fpga_session, args.fpga_stream_session).mainloop()


if __name__ == '__main__':
    main()
