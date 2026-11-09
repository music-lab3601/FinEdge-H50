"""Readable market dashboard: observed prices, live prediction and one review."""
from collections import OrderedDict, deque
import tkinter as tk
from tkinter import ttk

from experiments.reference.runtime.market import CandleSeries, INTERVALS, exchange_time, segment_key
from shared.model_paths import CLASS_CN
from experiments.reference.runtime.ui_widgets import BG, PANEL, CARD, TEXT, MUTED, ACCENT, UP, DOWN, SELECT, REALIZED, ValueCard, direction, direction_color


def npu_output(outputs):
    return outputs.get('board_socket_npu', outputs.get('icraft_socket'))


class MarketView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.series = {label: CandleSeries(ns) for label, ns in INTERVALS.items()}
        self.observations = deque(maxlen=20000)
        self.predictions = OrderedDict()
        self.selected = self.latest_resolved = self.latest_prediction = None
        self.dirty = True
        self.hitboxes = []
        self.on_follow = lambda: None
        self.interval = tk.StringVar(value='100毫秒')
        self.chart_kind = tk.StringVar(value='中间价走势')
        self.visible_bars = tk.StringVar(value='60')
        self.market_text = tk.StringVar(value='AAPL · 等待行情')
        self.price_text = tk.StringVar(value='— USD')
        self.market_clock = tk.StringVar(value='历史行情回放 · 时间为纽约交易所时间')
        self.accuracy_text = tk.StringVar(value='尚无已核对预测')
        self.selection_text = tk.StringVar(value='等待预测')
        self.review_title = tk.StringVar(value='03  预测复盘 · 先运行，再点击下表中的一行')
        self.hover_text = tk.StringVar(value='中间价 =（最优买价＋最优卖价）÷ 2；这是盘口价格，不是成交价。')
        self.agreement_text = tk.StringVar(value='先积累100条盘口，再发出第一条预测')
        self.score_cpu = tk.StringVar(value='CPU 命中  —')
        self.score_npu = tk.StringVar(value='NPU 命中  —')
        self.score_count = tk.StringVar(value='已核对 0 · 等待 0')
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, minsize=238)
        self.rowconfigure(0, weight=1)

        chart_panel = ttk.Frame(self)
        chart_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
        quote = ttk.Frame(chart_panel)
        quote.pack(fill='x')
        ttk.Label(quote, text='01  真实行情', style='Section.TLabel').pack(side='left')
        ttk.Label(quote, textvariable=self.price_text, style='Price.TLabel').pack(side='right')
        ttk.Label(chart_panel, textvariable=self.market_text, foreground=MUTED).pack(anchor='w')
        controls = ttk.Frame(chart_panel)
        controls.pack(fill='x', pady=(4, 3))
        for variable, values, width in ((self.chart_kind, ('中间价走势', 'K线'), 12),
                                         (self.interval, tuple(INTERVALS), 9),
                                         (self.visible_bars, ('30', '60', '120'), 5)):
            box = ttk.Combobox(controls, textvariable=variable, values=values, width=width, state='readonly')
            box.pack(side='left', padx=(0, 6))
            box.bind('<<ComboboxSelected>>', lambda _: self.invalidate())
        ttk.Label(controls, text='个时间桶 · 仅已回放部分', foreground=MUTED).pack(side='left')
        self.canvas = tk.Canvas(chart_panel, bg=PANEL, height=225, highlightthickness=1,
                                highlightbackground='#263b56')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _: self.invalidate())
        self.canvas.bind('<Motion>', self.hover)
        ttk.Label(chart_panel, text='A 金色：预测发生时刻     B 青色：50条后的实际结果     红涨 / 绿跌',
                  foreground=MUTED, font=('Microsoft YaHei UI', 9)).pack(anchor='w', pady=(3, 0))
        ttk.Label(chart_panel, textvariable=self.hover_text, foreground=MUTED,
                  font=('Microsoft YaHei UI', 9), wraplength=760).pack(anchor='w')

        sidebar = ttk.Frame(self)
        sidebar.grid(row=0, column=1, sticky='nsew')
        ttk.Label(sidebar, text='02  最新预测', style='Section.TLabel').pack(anchor='w')
        self.sidebar_hint = ttk.Label(sidebar, text='目标：再过50条盘口后的方向', foreground=MUTED,
                                      font=('Microsoft YaHei UI', 9))
        self.sidebar_hint.pack(anchor='w', pady=(0, 5))
        self.live_cpu = ValueCard(sidebar, 'CPU 参考', '等待预测', '窗口尚未形成', size=18, compact=True)
        self.live_npu = ValueCard(sidebar, '板载 NPU', '等待预测', '尚未执行', size=18, compact=True)
        self.live_cpu.pack(fill='x', pady=(0, 6))
        self.live_npu.pack(fill='x')
        ttk.Label(sidebar, textvariable=self.agreement_text, wraplength=230,
                  font=('Microsoft YaHei UI', 9)).pack(anchor='w', pady=6)
        self.score_panel = ttk.Frame(sidebar)
        self.score_panel.pack(fill='x')
        ttk.Separator(self.score_panel).pack(fill='x', pady=(0, 5))
        for var in (self.score_cpu, self.score_npu, self.score_count):
            ttk.Label(self.score_panel, textvariable=var, font=('Microsoft YaHei UI', 9)).pack(anchor='w')

        review = ttk.Frame(self)
        review.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        review_header = ttk.Frame(review)
        review_header.pack(fill='x', pady=(0, 5))
        ttk.Label(review_header, textvariable=self.review_title, style='Section.TLabel').pack(side='left')
        ttk.Button(review_header, text='跟随最新已核对窗口', command=self.follow).pack(side='right')
        cards = ttk.Frame(review)
        cards.pack(fill='x')
        for col in range(3):
            cards.columnconfigure(col, weight=1, uniform='review')
        self.origin_card = ValueCard(cards, 'A  当时的真实价格', '等待数据', '选一条预测，查看它的起点', size=20)
        self.forecast_card = ValueCard(cards, '模型当时的判断', '等待预测', '只预测方向，不预测具体价格', size=17)
        self.actual_card = ValueCard(cards, 'B  50条后的实际结果', '尚未到达', '未来数据到齐后，才判断对错', size=20)
        for col, card in enumerate((self.origin_card, self.forecast_card, self.actual_card)):
            card.grid(row=0, column=col, sticky='nsew', padx=(0 if col == 0 else 5, 0 if col == 2 else 5))

    def set_compact(self, compact):
        if compact:
            self.score_panel.pack_forget()
            self.sidebar_hint.pack_forget()
        else:
            self.sidebar_hint.pack(before=self.live_cpu, anchor='w', pady=(0, 5))
            self.score_panel.pack(fill='x')

    def invalidate(self):
        self.dirty = True

    def clear(self, new_run=False):
        for series in self.series.values():
            series.bars.clear()
        self.observations.clear()
        self.latest_resolved = self.selected = self.latest_prediction = None
        self.dirty = True
        self.live_cpu.set('积累窗口中', '满100条盘口后开始预测', MUTED)
        self.live_npu.set('等待执行', '只在启用板卡模式时运行', MUTED)
        self.agreement_text.set('先积累100条盘口，再发出第一条预测')
        self.origin_card.set('等待数据', '选一条预测，查看它的起点')
        self.forecast_card.set('等待预测', '只预测方向，不预测具体价格')
        self.actual_card.set('尚未到达', '未来数据到齐后，才判断对错', MUTED)
        self.review_title.set('03  预测复盘 · 等待新窗口')
        if new_run:
            self.predictions.clear()
            self.update_evaluation({})
        self.selection_text.set('等待预测；H50是50条盘口，不是50秒。')

    def follow(self):
        self.selected = None
        self.on_follow()
        self.dirty = True

    def select(self, window_id):
        self.selected = window_id
        self.dirty = True

    def apply_snapshot(self, snapshot):
        """Replace display state already aggregated from every source event."""
        for label, series in self.series.items():
            series.bars.clear()
            series.bars.extend(snapshot['series'].get(label, []))
        self.observations.clear()
        self.observations.extend(snapshot['observations'])
        rows = snapshot['recent_predictions']
        if rows:
            self.consume({'type': 'prediction', **rows[-1], 'evaluation': snapshot['evaluation']})
        self.predictions = OrderedDict((row['window_id'], row) for row in rows)
        self.latest_prediction = snapshot['latest_prediction_id']
        self.latest_resolved = snapshot['latest_resolved_id']
        if self.selected not in self.predictions:
            self.selected = None
        if self.latest_resolved not in self.predictions:
            self.latest_resolved = None
        market = snapshot.get('market')
        if market:
            meta = market['last']
            self.price_text.set(f"${market['mid2']/20000:.5f}")
            self.market_text.set(f"{meta['symbol']}  ·  {meta['date']}  {exchange_time(meta['timestamp_ns_since_exchange_midnight'])} 纽约  ·  历史盘口回放")
        self.update_evaluation(snapshot['evaluation'])
        self.dirty = True

    def consume(self, item):
        kind = item['type']
        if kind == 'market':
            bars = self.series[self.interval.get()].bars
            if bars and segment_key(item['last'], item['round']) != bars[-1]['segment_key']:
                self.clear()
            for series in self.series.values():
                series.push(item['last'], item['mid2'], item['round'])
            meta = item['last']
            self.observations.append((meta['timestamp_ns_since_exchange_midnight'], item['mid2']))
            self.price_text.set(f"${item['mid2']/20000:.5f}")
            self.market_text.set(f"{meta['symbol']}  ·  {meta['date']}  {exchange_time(meta['timestamp_ns_since_exchange_midnight'])} 纽约  ·  历史盘口回放")
        elif kind == 'prediction':
            self.predictions[item['window_id']] = item
            self.latest_prediction = item['window_id']
            while len(self.predictions) > 2000:
                self.predictions.popitem(last=False)
            for name, card in (('torch_cpu', self.live_cpu), ('icraft_socket', self.live_npu)):
                result = npu_output(item['outputs']) if name == 'icraft_socket' else item['outputs'].get(name)
                cls = result['class_index'] if result else None
                card.set(direction(cls), f"窗口 #{item['window_id']} · {result['elapsed_ns']/1e6:.2f} ms" if result else '当前模式未启用NPU', direction_color(cls))
            cpu = item['outputs']['torch_cpu']['class_index']
            npu = (npu_output(item['outputs']) or {}).get('class_index')
            self.agreement_text.set('CPU / NPU 方向一致，是否预测正确仍需等待实际结果' if npu == cpu else
                                    'CPU / NPU 方向不同，需查看工程详情' if npu is not None else 'CPU单路预测；未执行板载NPU')
            self.update_evaluation(item['evaluation'])
        elif kind == 'outcome':
            if item['window_id'] in self.predictions:
                self.predictions[item['window_id']]['outcome'] = item
            if item['status'] == 'resolved':
                self.latest_resolved = item['window_id']
            self.update_evaluation(item['evaluation'])
        elif kind == 'round':
            self.clear()
        elif kind == 'finished':
            self.update_evaluation(item['summary'].get('market_evaluation', {}))
        self.dirty = True

    def update_evaluation(self, summary):
        scores = []
        for name, label, var in (('torch_cpu', 'CPU', self.score_cpu), ('icraft_socket', 'NPU', self.score_npu)):
            backends = summary.get('backends', {})
            metric = npu_output(backends) if name == 'icraft_socket' else backends.get(name)
            text = f"{label} 命中 {metric['correct']}/{metric['count']}  ·  {metric['accuracy']:.1%}" if metric else f'{label} 命中 —'
            var.set(text)
            scores.append(text)
        self.score_count.set(f"已核对 {summary.get('resolved', 0)} · 等待 {summary.get('pending', 0)} · 未兑现 {summary.get('unresolved', 0)}")
        self.accuracy_text.set(' · '.join(scores)+' · '+self.score_count.get())

    def review_prediction(self):
        key = self.selected or self.latest_resolved or self.latest_prediction
        prediction = self.predictions.get(key)
        if not prediction:
            return None
        meta = prediction['last']
        outputs, outcome = prediction['outputs'], prediction.get('outcome', {})
        cpu = outputs['torch_cpu']['class_index']
        npu = (npu_output(outputs) or {}).get('class_index')
        self.review_title.set(f"03  预测复盘 · 窗口 #{key} · 第{prediction['round']}轮 · " + ('手动选中' if self.selected else '自动跟随'))
        self.origin_card.set(f"${prediction['origin_mid2']/20000:.5f}",
                             exchange_time(meta['timestamp_ns_since_exchange_midnight'])+' 纽约 · 预测起点 A', SELECT)
        self.forecast_card.set(f"CPU {CLASS_CN[cpu]}  /  NPU {CLASS_CN[npu] if npu is not None else '未启用'}",
                               '判断的是后面第50条盘口的价格方向')
        description = f"窗口#{key} · CPU{CLASS_CN[cpu]} / NPU{CLASS_CN[npu] if npu is not None else '—'}"
        if outcome.get('status') == 'resolved':
            actual = outcome['actual_class']
            actual_time = outcome['actual_meta']['timestamp_ns_since_exchange_midnight']
            verdicts = []
            for name, label in (('torch_cpu', 'CPU'), ('icraft_socket', 'NPU')):
                actual_name = 'board_socket_npu' if name == 'icraft_socket' and 'board_socket_npu' in outcome['matches'] else name
                if actual_name in outcome['matches']:
                    verdicts.append(label+('命中' if outcome['matches'][actual_name] else '未命中'))
            change = (outcome['actual_mid2']-prediction['origin_mid2'])/20000
            self.actual_card.set(('↓ 实际下跌', '→ 实际持平', '↑ 实际上涨')[actual],
                f"${outcome['actual_mid2']/20000:.5f}（{change:+.5f}） · {exchange_time(actual_time)}\n"+' / '.join(verdicts), direction_color(actual))
            description += f" · 真实{CLASS_CN[actual]} · "+' / '.join(verdicts)
        elif outcome:
            self.actual_card.set('未兑现', '停止、重置或片段结束，未来不足50条\n此窗口不计入命中率', MUTED)
            description += ' · 未兑现'
        else:
            self.actual_card.set('等待未来50条', '还没有真实结果，暂不判断对错', MUTED)
            description += ' · 等待真实结果'
        self.selection_text.set(description)
        return prediction

    def draw(self):
        self.dirty = False
        c = self.canvas
        c.delete('all')
        self.hitboxes = []
        w, h = max(c.winfo_width(), 300), max(c.winfo_height(), 90)
        prediction = self.review_prediction()
        bars = list(self.series[self.interval.get()].bars)[-int(self.visible_bars.get()):]
        if not bars:
            c.create_text(w/2, h/2-12, text='点击「开始回放」，观察真实盘口走势', fill=TEXT, font=('Microsoft YaHei UI', 13))
            c.create_text(w/2, h/2+15, text='前99条积累窗口，第100条产生第一条预测', fill=MUTED, font=('Microsoft YaHei UI', 10))
            return
        interval = INTERVALS[self.interval.get()]
        low, high = min(b['low_mid2'] for b in bars), max(b['high_mid2'] for b in bars)
        span = max(high-low, 4)
        low, high = low-span*.18, high+span*.18
        left, right, top, bottom = 18, w-87, 24, h-32
        y = lambda value: bottom-(value-low)/(high-low)*(bottom-top)
        first_time, last_time = bars[0]['bucket_ns'], bars[-1]['bucket_ns']+interval
        x = lambda timestamp: left+(timestamp-first_time)/(last_time-first_time)*(right-left)
        for fraction in (0, .25, .5, .75, 1):
            value = low+(high-low)*fraction
            yy = y(value)
            c.create_line(left, yy, right, yy, fill='#22344b')
            c.create_text(right+9, yy, text=f'{value/20000:.5f}', fill=MUTED, anchor='w', font=('Consolas', 9))
        for timestamp, anchor in ((first_time, 'w'), ((first_time+last_time)//2, 'center'), (last_time, 'e')):
            c.create_text(x(timestamp), h-12, text=exchange_time(timestamp), fill=MUTED, anchor=anchor, font=('Consolas', 9))
        for bar in bars:
            self.hitboxes.append((x(bar['bucket_ns']+interval/2), bar))
        if self.chart_kind.get() == 'K线':
            width = max(2, min(32, (right-left)*interval/(last_time-first_time)*.64))
            for center, bar in self.hitboxes:
                color = UP if bar['close_mid2'] > bar['open_mid2'] else DOWN if bar['close_mid2'] < bar['open_mid2'] else MUTED
                c.create_line(center, y(bar['high_mid2']), center, y(bar['low_mid2']), fill=color, width=1.5, tags='candle')
                a, b = sorted((y(bar['open_mid2']), y(bar['close_mid2'])))
                c.create_rectangle(center-width/2, a, center+width/2, max(a+2, b),
                                   fill=color if bar['closed'] else PANEL, outline=color,
                                   dash=() if bar['closed'] else (3, 2), tags='candle')
        else:
            points = [(x(t), y(mid)) for t, mid in self.observations if first_time <= t < last_time]
            if len(points) > 1:
                flat = [v for pair in points for v in pair]
                c.create_polygon(points[0][0], bottom, *flat, points[-1][0], bottom,
                                 fill='#172d48', outline='', tags='price_area')
                c.create_line(*flat, fill=ACCENT, width=2, tags='price_line')
            elif points:
                xx, yy = points[0]
                c.create_oval(xx-3, yy-3, xx+3, yy+3, fill=ACCENT, outline='')
        current = bars[-1]['close_mid2']
        yy = y(current)
        c.create_line(left, yy, right, yy, fill='#4a6c96', dash=(3, 5))
        c.create_rectangle(right+3, yy-10, w-3, yy+10, fill='#264d79', outline='')
        c.create_text(right+8, yy, text=f'{current/20000:.5f}', fill=TEXT, anchor='w', font=('Consolas', 9, 'bold'))
        if not prediction:
            return
        meta = prediction['last']
        timestamp = meta['timestamp_ns_since_exchange_midnight']
        if segment_key(meta, prediction['round']) != bars[-1]['segment_key'] or not first_time <= timestamp < last_time:
            c.create_text(left+8, top, text='所选窗口超出当前视野，可点“跟随最新已核对窗口”', fill=SELECT, anchor='w', font=('Microsoft YaHei UI', 9))
            return
        xx, yy = x(timestamp), y(prediction['origin_mid2'])
        c.create_line(xx, top, xx, bottom, fill=SELECT, dash=(4, 4), tags='selected')
        c.create_oval(xx-5, yy-5, xx+5, yy+5, fill=PANEL, outline=SELECT, width=2, tags='selected')
        c.create_text(xx-9, max(top, yy-15), text='A', fill=SELECT, font=('Segoe UI', 11, 'bold'), tags='selected')
        outcome = prediction.get('outcome', {})
        if outcome.get('status') == 'resolved':
            future_time = outcome['actual_meta']['timestamp_ns_since_exchange_midnight']
            if first_time <= future_time < last_time:
                fx, fy = x(future_time), y(outcome['actual_mid2'])
                c.create_line(xx, yy, fx, fy, fill=REALIZED, width=2, dash=(5, 3), tags='realized')
                c.create_oval(fx-5, fy-5, fx+5, fy+5, fill=REALIZED, outline=TEXT, tags='realized')
                c.create_text(min(right-5, fx+12), max(top, fy-16), text='B', fill=REALIZED,
                              font=('Segoe UI', 11, 'bold'), tags='realized')

    def hover(self, event):
        if not self.hitboxes:
            return
        _, bar = min(self.hitboxes, key=lambda pair: abs(pair[0]-event.x))
        prices = ' / '.join(f"{bar[key]/20000:.5f}" for key in ('open_mid2', 'high_mid2', 'low_mid2', 'close_mid2'))
        self.hover_text.set(f"{exchange_time(bar['bucket_ns'])} · 开/高/低/收 ${prices} · {bar['samples']}条盘口 · " +
                            ('已封存' if bar['closed'] else '形成中'))
