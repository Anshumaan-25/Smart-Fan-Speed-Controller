"""Desktop demonstration for the Smart Fan Controller project."""

import math
import sys
import tkinter as tk
from tkinter import ttk

from controllers import (
    EPOCHS, HELD_OUT_CASES, HUMIDITY_LABELS, RATES, RULES, SPEED_LABELS,
    TEMP_LABELS, TEMP_MAX, TEMP_MIN, TEMP_THRESHOLDS, TRAINING_CASES, crisp_speed,
    fuzzy_inference, reference_speed, run_experiment,
)

BG = "#0a111d"
PANEL = "#121d2e"
SURFACE = "#1b2a40"
GRID = "#31435c"
TEXT = "#f2f6ff"
MUTED = "#aabbd2"
AMBER, TEAL, VIOLET = "#ffba75", "#57e0c0", "#b5aaff"
COLORS = (AMBER, TEAL, VIOLET)
METHODS = ("Crisp thresholds", "Fuzzy controller", "Neural network")
SCENARIOS = (
    ("29.9 °C", 29.9, 60), ("30.1 °C", 30.1, 60),
    ("Cool / dry", 21, 25), ("Hot / humid", 37, 85),
)


def text(parent, value, size=12, color=TEXT, bg=PANEL, bold=False, **options):
    return tk.Label(
        parent, text=value, font=("Helvetica", size, "bold" if bold else "normal"),
        fg=color, bg=bg, **options,
    )


def canvas(parent, height, bg=PANEL):
    return tk.Canvas(parent, width=1, height=height, bg=bg, highlightthickness=0)


class FanLab:
    def __init__(self, root):
        self.root = root
        self.models, self.losses, self.errors = run_experiment()
        self.temperature = tk.DoubleVar(value=29.9)
        self.humidity = tk.IntVar(value=60)
        self.rate = tk.IntVar(value=1)
        self.fan_source = tk.StringVar(value=METHODS[1])
        self.motion = True
        self.angle = 0
        self.response_key = None
        self.chart_job = None

        root.title("Smart Fan Controller · Soft Computing Lab")
        root.configure(bg=BG)
        root.geometry("1050x700")
        root.minsize(890, 640)
        self.configure_styles()

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=24, pady=(15, 12))
        text(header, "SOFT COMPUTING LAB   /   MODULES 01–03", 10, TEAL, BG, True).pack(anchor="w")
        text(header, "Smart Fan Speed Controller", 27, bg=BG, bold=True).pack(side="left")
        text(header, "ROOM SIMULATION  •  SYNTHETIC DATA", 10, MUTED, BG, True).pack(side="right", pady=(8, 0))

        workspace = tk.Frame(root, bg=BG)
        workspace.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        self.build_controls(workspace)
        main = tk.Frame(workspace, bg=BG)
        main.pack(side="left", fill="both", expand=True, padx=(14, 0))
        self.build_results(main)
        self.build_analysis(main)

        for variable in (self.temperature, self.humidity, self.rate):
            variable.trace_add("write", self.refresh)
        self.fan_source.trace_add("write", self.update_fan_readout)
        self.refresh()
        self.animate_fan()

    def configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Lab.TButton", background=SURFACE, foreground=TEXT, bordercolor=GRID,
                        lightcolor=SURFACE, darkcolor=SURFACE, focuscolor=TEAL, padding=(7, 4))
        style.map("Lab.TButton", background=[("pressed", GRID), ("active", GRID)])
        style.configure("Lab.TCombobox", fieldbackground=SURFACE, foreground=TEXT,
                        background=SURFACE, arrowcolor=TEAL, bordercolor=GRID,
                        lightcolor=SURFACE, darkcolor=SURFACE, padding=4)
        style.map("Lab.TCombobox", fieldbackground=[("readonly", SURFACE)],
                  foreground=[("readonly", TEXT)], selectbackground=[("readonly", SURFACE)],
                  selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", SURFACE)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        style.configure("Lab.TNotebook", background=BG, bordercolor=GRID,
                        lightcolor=PANEL, darkcolor=PANEL, borderwidth=1)
        style.configure("Lab.TNotebook.Tab", background=BG, foreground=MUTED,
                        lightcolor=BG, darkcolor=BG, bordercolor=GRID,
                        focuscolor=TEAL, padding=(14, 8), borderwidth=1)
        style.map("Lab.TNotebook.Tab", background=[("selected", PANEL), ("active", SURFACE)],
                  foreground=[("selected", TEXT)], lightcolor=[("selected", PANEL)],
                  darkcolor=[("selected", PANEL)])

    def build_controls(self, parent):
        controls = tk.Frame(parent, bg=PANEL, padx=14, pady=12, width=236)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)
        text(controls, "Room conditions", 16, bold=True).pack(anchor="w")
        self.readouts = []
        for name, low, high, step, variable in (
            ("Temperature", TEMP_MIN, TEMP_MAX, 0.1, self.temperature),
            ("Relative humidity", 0, 100, 1, self.humidity),
        ):
            row = tk.Frame(controls, bg=PANEL)
            row.pack(fill="x", pady=(9, 0))
            text(row, name, 11, MUTED).pack(side="left")
            value = text(row, "", 12, TEAL, bold=True)
            value.pack(side="right")
            self.readouts.append(value)
            tk.Scale(
                controls, from_=low, to=high, resolution=step, variable=variable,
                orient="horizontal", showvalue=False, bg=PANEL, fg=TEXT,
                troughcolor=GRID, activebackground=TEAL, highlightbackground=PANEL,
                highlightcolor=TEAL, highlightthickness=1, borderwidth=0,
            ).pack(fill="x")

        text(controls, "DEMO PRESETS", 10, MUTED, bold=True).pack(anchor="w", pady=(12, 4))
        presets = tk.Frame(controls, bg=PANEL)
        presets.pack(fill="x")
        presets.columnconfigure((0, 1), weight=1, uniform="presets")
        for index, (name, temperature, humidity) in enumerate(SCENARIOS):
            ttk.Button(
                presets, text=name, style="Lab.TButton",
                command=lambda t=temperature, h=humidity: self.preset(t, h),
            ).grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)

        tk.Frame(controls, bg=GRID, height=1).pack(fill="x", pady=(13, 10))
        text(controls, "Fan preview", 16, bold=True).pack(anchor="w")
        ttk.Combobox(
            controls, state="readonly", textvariable=self.fan_source, values=METHODS,
            style="Lab.TCombobox", width=16,
        ).pack(fill="x", pady=(6, 0))
        self.fan = canvas(controls, 136)
        self.fan.pack(fill="both", expand=True)
        self.fan.bind("<Configure>", self.draw_fan)
        row = tk.Frame(controls, bg=PANEL)
        row.pack(fill="x")
        self.fan_speed = text(row, "", 18, bold=True)
        self.fan_speed.pack(side="left")
        self.motion_button = ttk.Button(row, text="Pause", width=6, style="Lab.TButton", command=self.toggle_motion)
        self.motion_button.pack(side="right")

    def build_results(self, parent):
        results = tk.Frame(parent, bg=BG)
        results.pack(fill="x", pady=(0, 11))
        results.columnconfigure((0, 1, 2), weight=1, uniform="methods")
        self.cards = []
        for column, (name, module, color) in enumerate(zip(METHODS, ("01", "03", "02"), COLORS)):
            card = tk.Frame(results, bg=PANEL, padx=12, pady=9)
            card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
            text(card, f"MODULE {module}", 10, color, bold=True).pack(anchor="w")
            text(card, name, 14, bold=True).pack(anchor="w")
            value = text(card, "", 25, color, bold=True)
            value.pack(anchor="w")
            meter = canvas(card, 5, SURFACE)
            meter.pack(fill="x", pady=(1, 4))
            meter.bind("<Configure>", lambda event, i=column: self.draw_meter(i))
            error = text(card, "", 11, MUTED)
            error.pack(anchor="w")
            self.cards.append((value, meter, error))

    def build_analysis(self, parent):
        self.tabs = ttk.Notebook(parent, style="Lab.TNotebook")
        self.tabs.pack(fill="both", expand=True)
        self.tabs.enable_traversal()
        self.panels = []
        for title in ("Temperature sweep", "Fuzzy rules", "Neural training"):
            panel = tk.Frame(self.tabs, bg=PANEL, padx=12, pady=8)
            self.tabs.add(panel, text=title)
            self.panels.append(panel)
        response, fuzzy, training = self.panels

        text(response, "Speed across the temperature range", 16, bold=True).pack(anchor="w")
        self.response_detail = text(response, "", 11, MUTED)
        self.response_detail.pack(anchor="w")
        self.legend(response, (
            ("Crisp", AMBER), ("Fuzzy", TEAL), ("Neural", VIOLET), ("Simulated target", MUTED),
        ))
        self.response_plot = canvas(response, 150)
        self.response_plot.pack(fill="both", expand=True)
        self.reference_detail = text(response, "", 11, MUTED)
        self.reference_detail.pack(anchor="w")

        text(fuzzy, "Mamdani inference", 16, bold=True).pack(anchor="w")
        text(fuzzy, "Each cell shows a rule output and its live firing strength.", 11, MUTED).pack(anchor="w")
        rules = tk.Frame(fuzzy, bg=PANEL)
        rules.pack(fill="x", pady=(4, 5))
        rules.columnconfigure((1, 2, 3), weight=1, uniform="rules")
        self.rule_cells = []
        for column, humidity in enumerate(HUMIDITY_LABELS, start=1):
            text(rules, humidity, 10, MUTED, bold=True).grid(row=0, column=column)
        for row, (temperature, outputs) in enumerate(zip(TEMP_LABELS, RULES), start=1):
            text(rules, temperature, 11, MUTED).grid(row=row, column=0, sticky="w", padx=(0, 8))
            cells = []
            for column, output in enumerate(outputs, start=1):
                cell = text(rules, SPEED_LABELS[output], 11, bg=SURFACE, pady=2)
                cell.grid(row=row, column=column, sticky="ew", padx=2, pady=2)
                cells.append(cell)
            self.rule_cells.append(cells)
        self.fuzzy_detail = text(fuzzy, "", 11, TEAL)
        self.fuzzy_detail.pack(anchor="w")
        self.fuzzy_plot = canvas(fuzzy, 115)
        self.fuzzy_plot.pack(fill="both", expand=True)

        row = tk.Frame(training, bg=PANEL)
        row.pack(fill="x")
        text(row, "Backpropagation training", 16, bold=True).pack(side="left")
        self.rate_picker = ttk.Combobox(
            row, state="readonly", values=[f"Learning rate {rate:g}" for rate in RATES],
            width=18, style="Lab.TCombobox",
        )
        self.rate_picker.current(self.rate.get())
        self.rate_picker.bind("<<ComboboxSelected>>", lambda _: self.rate.set(self.rate_picker.current()))
        self.rate_picker.pack(side="right")
        text(
            training, f"2–6–1 MLP   •   {len(TRAINING_CASES)} training cases   •   {EPOCHS} epochs",
            11, MUTED,
        ).pack(anchor="w")
        self.legend(training, tuple((f"η = {rate:g}", color) for rate, color in zip(RATES, (AMBER, VIOLET))))
        self.training_plot = canvas(training, 150)
        self.training_plot.pack(fill="both", expand=True)
        self.training_detail = text(training, "", 11, MUTED)
        self.training_detail.pack(anchor="w")

        for plot in (self.response_plot, self.fuzzy_plot, self.training_plot):
            plot.bind("<Configure>", self.queue_chart)
        self.tabs.bind("<<NotebookTabChanged>>", self.queue_chart)
        text(
            parent, f"HELD-OUT MAE  /  {len(HELD_OUT_CASES)} SIMULATED CASES  /  PP = PERCENTAGE POINTS",
            9, MUTED, BG, True,
        ).pack(anchor="w", pady=(8, 0))

    def legend(self, parent, entries):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(4, 0))
        for name, color in entries:
            text(row, f"●  {name}", 11, color).pack(side="left", padx=(0, 14))

    def preset(self, temperature, humidity=60):
        self.temperature.set(temperature)
        self.humidity.set(humidity)

    def refresh(self, *_):
        temperature, humidity = self.temperature.get(), self.humidity.get()
        index = self.rate.get()
        self.rate_picker.current(index)
        self.readouts[0].configure(text=f"{temperature:.1f} °C")
        self.readouts[1].configure(text=f"{humidity}%")
        self.fuzzy = fuzzy_inference(temperature, humidity)
        self.speeds = (crisp_speed(temperature, humidity), self.fuzzy.speed,
                       self.models[index].speed(temperature, humidity))
        for column, ((value, _, error), speed, mae) in enumerate(zip(
            self.cards, self.speeds, (*self.errors[:2], self.errors[2 + index])
        )):
            value.configure(text=f"{speed:.1f}%")
            error.configure(text=f"Held-out MAE  {mae:.2f} pp")
            self.draw_meter(column)
        self.update_fan_readout()
        self.response_detail.configure(text=f"Humidity fixed at {humidity}%  •  Marker at {temperature:.1f} °C")
        self.reference_detail.configure(
            text=f"Simulated target at marker  {reference_speed(temperature, humidity):.1f}%"
        )
        strongest = self.fuzzy.strongest[0]
        for cells, outputs, strengths in zip(self.rule_cells, RULES, self.fuzzy.firing):
            for cell, output, strength in zip(cells, outputs, strengths):
                cell.configure(text=f"{SPEED_LABELS[output]}  ·  {strength:.2f}",
                               fg=TEAL if strength > 0 else MUTED,
                               bg="#204345" if strength == strongest else SURFACE)
        self.fuzzy_detail.configure(text=f"Aggregated output  /  Centroid at {self.fuzzy.speed:.1f}%")
        history = self.losses[index]
        self.training_detail.configure(
            text=f"η = {RATES[index]:g}  •  MSE at epoch 0: {history[0]:.3g}"
                 f"  →  epoch {EPOCHS}: {history[-1]:.3g}"
        )
        self.queue_chart()

    def draw_meter(self, index):
        _, meter, _ = self.cards[index]
        meter.delete("all")
        meter.create_rectangle(0, 0, meter.winfo_width() * self.speeds[index] / 100, 5,
                               fill=COLORS[index], width=0)

    def update_fan_readout(self, *_):
        self.fan_index = METHODS.index(self.fan_source.get())
        self.fan_speed.configure(text=f"{self.speeds[self.fan_index]:.1f}%", fg=COLORS[self.fan_index])
        self.draw_fan()

    def toggle_motion(self):
        self.motion = not self.motion
        self.motion_button.configure(text="Pause" if self.motion else "Play")

    def animate_fan(self):
        if self.motion and self.fan.winfo_ismapped():
            self.angle = (self.angle + 0.0035 * self.speeds[self.fan_index]) % math.tau
            self.draw_fan()
        self.root.after(40, self.animate_fan)

    def draw_fan(self, _=None):
        plot = self.fan
        plot.delete("all")
        x, y = plot.winfo_width() / 2, plot.winfo_height() / 2
        radius = min(x, y, 86) - 5
        if radius < 16:
            return
        for factor, outline in ((1, GRID), (0.89, SURFACE), (0.68, GRID)):
            r = radius * factor
            plot.create_oval(x-r, y-r, x+r, y+r, outline=outline, width=2)
        color = COLORS[self.fan_index]
        for blade in range(4):
            rotation = self.angle + blade * math.pi / 2
            points = []
            for distance, offset in ((0.13, 0), (0.47, -0.27), (0.77, -0.06),
                                     (0.81, 0.43), (0.54, 0.64), (0.22, 0.32)):
                points.extend((x + radius * distance * math.cos(rotation + offset),
                               y + radius * distance * math.sin(rotation + offset)))
            plot.create_polygon(*points, fill=color, smooth=True, splinesteps=12)
        hub = radius * 0.14
        plot.create_oval(x-hub, y-hub, x+hub, y+hub, fill=PANEL, outline=color, width=2)

    def queue_chart(self, _=None):
        if self.chart_job is None:
            self.chart_job = self.root.after_idle(self.draw_active_chart)

    def draw_active_chart(self):
        self.chart_job = None
        tab = self.tabs.index("current")
        (self.draw_response, self.draw_fuzzy, self.draw_training)[tab]()

    def draw_response(self):
        plot = self.response_plot
        axes = Axes(plot, (TEMP_MIN, TEMP_MAX), (0, 100), range(18, 43, 4),
                    range(0, 101, 25), "Temperature (°C)", "%")
        if not axes.ready:
            return
        humidity, rate = self.humidity.get(), self.rate.get()
        if self.response_key != (humidity, rate):
            temperatures = [i / 4 for i in range(TEMP_MIN * 4, TEMP_MAX * 4 + 1)]
            cuts = (TEMP_MIN, *TEMP_THRESHOLDS, TEMP_MAX)
            crisp_steps = []
            for start, end in zip(cuts, cuts[1:]):
                speed = crisp_speed(start, humidity)
                crisp_steps.extend(((start, speed), (end, speed)))
            self.response_curves = (
                [(t, reference_speed(t, humidity)) for t in temperatures],
                crisp_steps,
                [(t, fuzzy_inference(t, humidity).speed) for t in temperatures],
                [(t, self.models[rate].speed(t, humidity)) for t in temperatures],
            )
            self.response_key = humidity, rate
        for points, color, dash in zip(self.response_curves, (MUTED, *COLORS), ((5, 4), (), (), ())):
            axes.line(points, color, dash)
        temperature = self.temperature.get()
        axes.marker(temperature, self.speeds, COLORS)

    def draw_fuzzy(self):
        axes = Axes(self.fuzzy_plot, (0, 100), (0, 1), range(0, 101, 25),
                    (0, 0.5, 1), "Fan speed (%)")
        if not axes.ready:
            return
        curve = list(enumerate(self.fuzzy.aggregate))
        coordinates = [coordinate for point in [(0, 0), *curve, (100, 0)] for coordinate in axes.point(*point)]
        self.fuzzy_plot.create_polygon(*coordinates, fill="#204345", outline="")
        axes.line(curve, TEAL)
        axes.marker(self.fuzzy.speed)

    def draw_training(self):
        logs = [[math.log10(max(loss, 1e-15)) for loss in history] for history in self.losses]
        low = math.floor(min(map(min, logs)))
        high = max(low + 1, math.ceil(max(map(max, logs))))
        axes = Axes(
            self.training_plot, (0, EPOCHS), (low, high), range(0, EPOCHS + 1, 25),
            range(low, high + 1), "Epoch  /  MSE on log scale", y_format=lambda value: f"1e{value}",
        )
        if not axes.ready:
            return
        for series, color, dash in zip(logs, (AMBER, VIOLET), ((5, 4), ())):
            axes.line(enumerate(series), color, dash)


class Axes:
    """Canvas plot coordinates shared by the three analysis views."""

    def __init__(self, plot, x_range, y_range, x_ticks, y_ticks, title, suffix="", y_format=None):
        self.plot = plot
        plot.delete("all")
        self.left, self.top = 48, 12
        self.right, self.bottom = plot.winfo_width() - 22, plot.winfo_height() - 34
        self.x_range, self.y_range = x_range, y_range
        self.ready = self.right > self.left and self.bottom > self.top
        if not self.ready:
            return
        font = ("Helvetica", 10)
        for value in y_ticks:
            _, y = self.point(x_range[0], value)
            plot.create_line(self.left, y, self.right, y, fill=GRID)
            label = y_format(value) if y_format else f"{value:g}{suffix}"
            plot.create_text(self.left - 7, y, anchor="e", text=label, fill=MUTED, font=font)
        for value in x_ticks:
            x, _ = self.point(value, y_range[0])
            plot.create_text(x, self.bottom + 9, text=str(value), fill=MUTED, font=font)
        plot.create_text((self.left + self.right) / 2, self.bottom + 24, text=title, fill=MUTED, font=font)

    def point(self, x, y):
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        return (self.left + (x - x_min) / (x_max - x_min) * (self.right - self.left),
                self.bottom - (y - y_min) / (y_max - y_min) * (self.bottom - self.top))

    def line(self, points, color, dash=()):
        coordinates = [coordinate for point in points for coordinate in self.point(*point)]
        self.plot.create_line(*coordinates, fill=color, width=2, dash=dash)

    def marker(self, value, readings=(), colors=()):
        x, _ = self.point(value, self.y_range[0])
        self.plot.create_line(x, self.top, x, self.bottom, fill=TEXT, dash=(3, 4))
        for reading, color in zip(readings, colors):
            _, y = self.point(value, reading)
            self.plot.create_oval(x-4, y-4, x+4, y+4, fill=color, outline=PANEL, width=2)


if __name__ == "__main__":
    if sys.platform == "darwin" and tk.TkVersion < 8.6:
        raise SystemExit("This app requires Tk 8.6 or newer on macOS. Run ./Launch-Fan-Lab.command.")
    root = tk.Tk()
    FanLab(root)
    root.mainloop()
