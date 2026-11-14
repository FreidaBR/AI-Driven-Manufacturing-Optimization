from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas


DOCS_DIR = Path(__file__).resolve().parent
PDF_NAME = "AI_Manufacturing_Optimization_Project_Overview.pdf"
PDF_OUTPUT = DOCS_DIR / PDF_NAME

NAVY = colors.HexColor("#12213A")
BLUE = colors.HexColor("#2563EB")
CYAN = colors.HexColor("#0891B2")
INK = colors.HexColor("#243247")
MUTED = colors.HexColor("#58677C")
PALE = colors.HexColor("#F2F6FC")
LINE = colors.HexColor("#D7E0EC")
GREEN = colors.HexColor("#087E67")
AMBER = colors.HexColor("#A85C05")
RED = colors.HexColor("#B42335")
WHITE = colors.white


class ReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_states = []

    def showPage(self):
        self.saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_total = len(self.saved_states)
        for state in self.saved_states:
            self.__dict__.update(state)
            self.draw_furniture(page_total)
            super().showPage()
        super().save()

    def draw_furniture(self, page_total: int):
        self.saveState()
        page_width, page_height = A4
        if self._pageNumber > 1:
            self.setFillColor(NAVY)
            self.setFont("Helvetica-Bold", 7.5)
            self.drawString(42, page_height - 25, "AI-DRIVEN MANUFACTURING OPTIMIZATION")
            self.setFillColor(MUTED)
            self.setFont("Helvetica", 7.5)
            self.drawRightString(page_width - 42, page_height - 25, "TECHNICAL PROJECT OVERVIEW")
            self.setStrokeColor(LINE)
            self.line(42, page_height - 32, page_width - 42, page_height - 32)
        self.setStrokeColor(LINE)
        self.line(42, 31, page_width - 42, 31)
        self.setFillColor(MUTED)
        self.setFont("Helvetica", 7.2)
        self.drawString(42, 19, "Historical-data-driven prototype | Evidence checked 14 November 2026")
        self.drawRightString(page_width - 42, 19, f"{self._pageNumber} / {page_total}")
        self.restoreState()


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 140

    def draw(self):
        box_width = (self.width - 5 * 10) / 6
        box_height = 56
        y = 62
        labels = [
            ("Historical", "telemetry +", "event CSVs"),
            ("Feature", "pipeline", "47 features"),
            ("Saved LightGBM", "failure / RUL /", "component"),
            ("Decision", "risk + priority +", "schedule"),
            ("FastAPI", "inference +", "replay API"),
            ("React", "dashboard +", "controls"),
        ]
        fills = ["#EAF1FF", "#EAF1FF", "#E7F7F5", "#FFF4E5", "#EAF1FF", "#EEF2F7"]
        for index, parts in enumerate(labels):
            x = index * (box_width + 10)
            self.canv.setFillColor(colors.HexColor(fills[index]))
            self.canv.setStrokeColor(LINE)
            self.canv.roundRect(x, y, box_width, box_height, 7, fill=1, stroke=1)
            self.canv.setFillColor(NAVY if index not in (2, 3) else GREEN if index == 2 else AMBER)
            self.canv.setFont("Helvetica-Bold", 7.2)
            self.canv.drawCentredString(x + box_width / 2, y + 39, parts[0])
            self.canv.setFillColor(INK)
            self.canv.setFont("Helvetica", 6.8)
            self.canv.drawCentredString(x + box_width / 2, y + 26, parts[1])
            self.canv.drawCentredString(x + box_width / 2, y + 15, parts[2])
            if index < len(labels) - 1:
                arrow_start = x + box_width + 1
                arrow_end = x + box_width + 8
                arrow_y = y + box_height / 2
                self.canv.setStrokeColor(BLUE)
                self.canv.setLineWidth(1.2)
                self.canv.line(arrow_start, arrow_y, arrow_end, arrow_y)
                self.canv.line(arrow_end, arrow_y, arrow_end - 3, arrow_y + 2.5)
                self.canv.line(arrow_end, arrow_y, arrow_end - 3, arrow_y - 2.5)
        self.canv.setFillColor(MUTED)
        self.canv.setFont("Helvetica-Oblique", 7)
        self.canv.drawCentredString(
            self.width / 2,
            37,
            "Replay cursor and snapshot are held in backend process memory; inputs remain historical records.",
        )
        self.canv.setFillColor(RED)
        self.canv.setFont("Helvetica-Bold", 7)
        self.canv.drawCentredString(
            self.width / 2,
            20,
            "No live sensor gateway, streaming broker, or CMMS integration is implemented.",
        )


def build_styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=30,
            alignment=TA_LEFT,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=BLUE,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "SubHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.4,
            leading=13,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyTextCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=12,
            textColor=INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallText",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=INK,
            spaceAfter=2,
        ),
        "cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.2,
            textColor=INK,
        ),
        "cell_bold": ParagraphStyle(
            "TableCellBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9.2,
            textColor=NAVY,
        ),
        "cell_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9,
            textColor=WHITE,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12.5,
            textColor=NAVY,
        ),
        "center": ParagraphStyle(
            "Centered",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            alignment=TA_CENTER,
            textColor=INK,
        ),
    }


def para(text: str, styles, style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def make_table(rows, widths, styles, *, header=True, font_size=None):
    converted = []
    for row_index, row in enumerate(rows):
        style = "cell_header" if header and row_index == 0 else "cell"
        converted.append([
            cell if isinstance(cell, Flowable) else para(str(cell), styles, style)
            for cell in row
        ])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
        commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]))
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, PALE]))
    table.setStyle(TableStyle(commands))
    return table


def add_page_title(story, number, title, subtitle, styles):
    story.append(para(f"{number}  {title}", styles, "h1"))
    if subtitle:
        story.append(para(subtitle, styles, "body"))


def create_document():
    styles = build_styles()
    page_width, _ = A4
    content_width = page_width - 84
    story = []

    # Page 1: Executive summary and scope.
    story.extend([
        Spacer(1, 24),
        para("AI-Driven Manufacturing<br/>Optimization", styles, "cover_title"),
        para(
            "Predictive maintenance using historical benchmark data",
            styles,
            "cover_subtitle",
        ),
    ])
    status = Table(
        [[para(
            "<b>PROJECT STATUS</b><br/>Historical-data-driven prototype",
            styles,
            "callout",
        )]],
        colWidths=[content_width],
    )
    status.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF1FF")),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([status, Spacer(1, 12)])

    cards = [
        [para("<b>100</b><br/>machines in benchmark", styles, "center"),
         para("<b>47</b><br/>model input features", styles, "center"),
         para("<b>24 h</b><br/>failure target horizon", styles, "center")],
        [para("<b>3</b><br/>saved LightGBM models", styles, "center"),
         para("<b>15 / 15</b><br/>backend tests passed", styles, "center"),
         para("<b>0</b><br/>live factory feeds", styles, "center")],
    ]
    card_table = Table(cards, colWidths=[content_width / 3] * 3, rowHeights=[48, 48])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("GRID", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([card_table, Spacer(1, 13)])

    add_page_title(story, "01", "Executive summary", None, styles)
    story.append(para(
        "This project is an educational predictive-maintenance prototype. It applies saved "
        "LightGBM models to engineered features from a historical manufacturing benchmark, "
        "then displays risk categories, machine rankings, advisory maintenance actions, and "
        "a daily-capacity-limited schedule proposal. A React dashboard controls a replay of "
        "the historical records through a FastAPI backend.",
        styles,
    ))
    story.append(para(
        "It is not a production deployment, safety system, or live-monitoring platform. "
        "The replay does not ingest new factory data. A schedule is a planning proposal, "
        "not a CMMS work order or assurance that maintenance can safely be completed before failure.",
        styles,
    ))
    story.append(para("Implemented scope", styles, "h2"))
    story.append(make_table([
        ["Capability", "What the checked-in implementation does"],
        ["Failure risk", "Predicts the probability of a recorded machine failure within the next 24 hours."],
        ["Risk and alerts", "Uses shared probability bands; an operational alert is raised at probability >= 0.98."],
        ["Historical replay", "Steps through historical sensor snapshots for 100 machines with start, pause, advance, reset, and preset scenarios."],
        ["Maintenance planning", "Ranks non-LOW proposals, applies a requested per-day capacity, and marks RUL-infeasible slots."],
    ], [content_width * 0.24, content_width * 0.76], styles))
    story.append(Spacer(1, 8))
    story.append(para(
        "<b>Important:</b> recommendations require qualified human review. Model risk limits "
        "are application configuration, not validated maintenance thresholds.",
        styles,
    ))
    story.append(PageBreak())

    # Page 2: Architecture and workflow.
    add_page_title(
        story,
        "02",
        "Architecture and workflow",
        "The data path shown below reflects the checked-in backend and frontend. Arrows describe software flow, not a live industrial control loop.",
        styles,
    )
    story.append(ArchitectureDiagram(content_width))
    story.append(para("Request and replay sequence", styles, "h2"))
    story.append(make_table([
        ["Step", "Implementation"],
        ["1. Load", "The backend validates saved model artifacts, the 47-feature schema, imputation statistics, and required datasets."],
        ["2. Engineer", "Telemetry, machine metadata, rolling sensor summaries, and strictly prior error/maintenance events are aligned to the exported feature order."],
        ["3. Infer", "The failure classifier, RUL regressor, and component classifier run on the historical snapshot."],
        ["4. Decide", "Probability bands determine category and alert status; the component/RUL outputs inform advisory maintenance details."],
        ["5. Present", "FastAPI returns snapshots and reports; React renders the dashboard and provides replay controls."],
    ], [content_width * 0.18, content_width * 0.82], styles))
    story.append(para("Software components", styles, "h2"))
    story.append(make_table([
        ["Layer", "Technology / role"],
        ["Inference and API", "Python, FastAPI, Uvicorn, Pydantic; model/data validation, inference, replay state, risk, and schedule endpoints."],
        ["Machine learning", "LightGBM classifier/regressor artifacts; scikit-learn and joblib support; NumPy and pandas for feature/data operations."],
        ["Dashboard", "React, Vite, and Recharts; replay controls, tables, visual summaries, diagnostics, and report modal."],
        ["Persistence", "Historical CSV/Parquet data and checked-in model artifacts. Replay position is process memory, not durable storage."],
    ], [content_width * 0.22, content_width * 0.78], styles))
    story.append(PageBreak())

    # Page 3: Model and threshold evidence.
    add_page_title(
        story,
        "03",
        "Model design and recorded evaluation",
        "Three saved LightGBM artifacts consume a common 47-feature input schema. The reported metrics are artifacts in the repository, not independently reproduced results in this report.",
        styles,
    )
    story.append(make_table([
        ["Model", "Target", "Recorded result"],
        ["Failure classifier", "Recorded failure in next 24 hours", "ROC-AUC 0.99989; PR-AUC 0.99266; F1 0.9762 at the recorded 0.98 threshold."],
        ["RUL regressor", "Hours to next recorded failure, capped at 168", "Urgent-window MAE 5.8 hours (as recorded in model metadata)."],
        ["Component classifier", "Failure component class (comp1-comp4)", "Accuracy 95.99% (as recorded in model metadata)."],
    ], [content_width * 0.22, content_width * 0.38, content_width * 0.40], styles))
    story.append(Spacer(1, 7))
    story.append(para(
        "Training/evaluation records describe a chronological holdout with a 24-hour "
        "embargo for the binary target. Existing reports disagree in some metric decimals "
        "and threshold counts; the values above quote model_metadata.json. The repository "
        "does not establish calibration, external-site generalization, or operational value. "
        "A Brier score in a separate candidate report is not, by itself, proof of calibrated "
        "failure probabilities.",
        styles,
    ))
    story.append(para("Backend probability bands and alert behavior", styles, "h2"))
    story.append(make_table([
        ["Predicted failure probability", "Risk category", "Maintenance label", "Alert"],
        ["p >= 0.98", "CRITICAL", "URGENT", "Yes"],
        ["0.85 <= p < 0.98", "HIGH", "PREVENTIVE", "No"],
        ["0.50 <= p < 0.85", "MEDIUM", "PREVENTIVE", "No"],
        ["p < 0.50", "LOW", "ROUTINE", "No"],
    ], [content_width * 0.31, content_width * 0.20, content_width * 0.25, content_width * 0.24], styles))
    story.append(Spacer(1, 7))
    story.append(para(
        "The alert condition is probability >= 0.98 in the backend configuration. The other "
        "bands categorize model scores but do not trigger the operational alert. These are "
        "project-defined thresholds, not certified plant or safety limits.",
        styles,
    ))
    story.append(para("Explanations and derived indicators", styles, "h2"))
    story.append(para(
        "The API exposes LightGBM contribution values in raw log-odds space; these describe "
        "model-score attribution and are not causal explanations. Sensor-baseline deviations "
        "and recent-error counts are separate heuristics. The displayed temperature is a "
        "formula-derived proxy from voltage, rotation, vibration, and pressure, not a separate "
        "temperature sensor channel. Missing model inputs use saved training medians and are "
        "shown as missing where relevant.",
        styles,
    ))
    story.append(PageBreak())

    # Page 4: Data and maintenance behavior.
    add_page_title(
        story,
        "04",
        "Data, replay, and maintenance planning",
        "Historical data provides the demonstration input; it is not evidence that a sensor gateway or online stream is present.",
        styles,
    )
    story.append(make_table([
        ["Data item", "Repository evidence and limitation"],
        ["Benchmark attribution", "README attributes the cleaned records to the Microsoft Azure Predictive Maintenance benchmark. No exact source version, acquisition record, citation URL, or license terms are established in the repository."],
        ["Telemetry", "876,100 telemetry CSV records are present for 100 machine IDs; timestamps and content remain historical benchmark data."],
        ["Features", "873,700 engineered Parquet rows are present; the saved input schema contains 47 features."],
        ["Events", "The checked-in failures table contains 761 records. Error and maintenance event tables and machine metadata are also checked in."],
        ["Derived values", "Temperature is calculated; downtime and some cost/duration figures use static assumptions rather than measured factory downtime."],
    ], [content_width * 0.22, content_width * 0.78], styles))
    story.append(para("Replay and dashboard behavior", styles, "h2"))
    story.append(para(
        "A replay step selects the next historical timestamp, scores the machines in that "
        "snapshot, and returns model-based priorities, alerts, and a proposed schedule. The "
        "demonstrated data covers 100 machines. The replay cursor is held in backend process "
        "memory: browser refresh can restore it while the backend process remains alive; "
        "restarting that process loses the cursor. Preset scenarios also point to historical "
        "positions rather than generating new telemetry.",
        styles,
    ))
    story.append(para("Capacity-constrained proposal logic", styles, "h2"))
    story.append(make_table([
        ["Rule", "Behavior"],
        ["Candidate set", "Non-LOW snapshot machines are sorted by failure probability, then predicted RUL, then machine ID."],
        ["Daily capacity", "The requested capacity is validated from 1 to 100 and proposal slots are allocated across days."],
        ["Feasibility", "The slot start plus assumed task duration is compared with predicted RUL; overdue/infeasible entries are labeled, not treated as safe actions."],
        ["Exclusions and assumptions", "A same-snapshot maintenance event is excluded. Stock, skill labels, costs, and task durations are illustrative; actual staff/parts availability and site constraints are not integrated."],
    ], [content_width * 0.23, content_width * 0.77], styles))
    story.append(Spacer(1, 6))
    story.append(para(
        "<b>Data governance:</b> confirm the benchmark's original source and terms before "
        "redistribution or operational use. This document does not assert a license.",
        styles,
    ))
    story.append(PageBreak())

    # Page 5: API and verification.
    add_page_title(
        story,
        "05",
        "API surface and verification",
        "The backend exposes the following representative application routes; see FastAPI /docs for the live OpenAPI schema.",
        styles,
    )
    story.append(make_table([
        ["Route", "Purpose"],
        ["GET /api/health; GET /api/model/health", "Data-file status, model load status, feature count, and risk configuration."],
        ["GET /api/predict/machine/{id}; GET /api/machines/{id}/details", "Machine-specific prediction and historical snapshot/details."],
        ["POST /api/predict", "Inference for a complete 47-feature vector."],
        ["GET /api/simulation/scenarios; POST /api/simulation/scenario/{id}", "List/load historical demonstration positions."],
        ["POST /api/simulation/step; GET /api/simulation/state; POST /api/simulation/reset", "Advance, read, and reset the in-memory historical replay."],
        ["GET /api/alerts/active; GET /api/maintenance/schedule_realistic", "Current replay alerts and capacity-limited proposal schedule."],
        ["GET /api/maintenance/report; GET /api/historical_trends", "Maintenance proposal report and 2015 historical event summaries."],
        ["GET /api/trial/downtime_reduction", "Assumption-based scenario comparison; not observed savings or an operational trial."],
    ], [content_width * 0.47, content_width * 0.53], styles))
    story.append(para("Current verification", styles, "h2"))
    story.append(make_table([
        ["Check", "Observed result"],
        ["Backend unittest suite", "15 passed, 0 failed, 0 skipped (run 14 November 2026)."],
        ["Model/data load", "Three model artifacts loaded; 47 features; required data present in the current environment."],
        ["API smoke checks", "Prediction, machine details, 100-machine replay, alert threshold, reset, and capacity-limited scheduling responded successfully; invalid timestamp returned 400 and invalid capacity returned 422."],
        ["Frontend", "Production build passed. Manual browser checks covered replay controls, refresh restoration, diagnostics, and report display; no automated browser E2E suite is included."],
        ["Environment caveat", "Test environment has scikit-learn 1.6.1 while requirements/artifacts specify 1.8.0; artifact loading emits an InconsistentVersionWarning. A clean install was not verified."],
    ], [content_width * 0.25, content_width * 0.75], styles))
    story.append(Spacer(1, 6))
    story.append(para(
        "The test count is the unittest result only. Manual smoke checks and the print-oriented "
        "verification script are not counted as additional automated tests.",
        styles,
    ))
    story.append(PageBreak())

    # Page 6: Limitations and future work.
    add_page_title(
        story,
        "06",
        "Limitations and future enhancements",
        "Items in the right column are proposed work, not implemented capabilities.",
        styles,
    )
    story.append(make_table([
        ["Area", "Current implementation / evidence boundary", "Proposed enhancement"],
        ["Connectivity", "No live factory sensor integration, message broker, or stream processing.", "Build a secure, timestamped ingestion path and validate it with a site-approved sensor source."],
        ["Maintenance operations", "No CMMS work-order integration, verified technician availability, actual inventory, or site safety constraints.", "Integrate with an authorized CMMS and enforce human approval and site procedures."],
        ["Model validity", "Recorded retrospective benchmark metrics; no independent plant validation or deployment calibration.", "Reproduce evaluation in a clean environment; assess calibration, drift, machine/site holdout, and prospective results."],
        ["Data and provenance", "Repository attribution is incomplete; license and redistribution terms are not verified.", "Record dataset version, source citation, terms, and transformation lineage."],
        ["Testing and scale", "15 backend unit tests and manual UI checks; no automated browser E2E suite or demonstrated scale beyond 100 machines.", "Add E2E, fault-injection, and load tests with measurable acceptance criteria."],
        ["Operational claims", "Downtime/cost comparisons use assumed durations and costs, not observed outcomes.", "Collect controlled, auditable operational outcomes before making savings claims."],
    ], [content_width * 0.19, content_width * 0.41, content_width * 0.40], styles))
    story.append(para("Claim verification checklist", styles, "h2"))
    checklist = [
        "Historical-data-driven prototype; not production-ready or safety-certified.",
        "24-hour failure classifier with 47-feature input schema and saved LightGBM model artifacts.",
        "Risk bands: CRITICAL >= 0.98, HIGH >= 0.85, MEDIUM >= 0.50, otherwise LOW; alerts at >= 0.98.",
        "Historical replay demonstrably covers the checked-in 100-machine dataset; replay is not live monitoring.",
        "Schedule capacity enforcement and explicit infeasibility status are implemented; operational feasibility is not guaranteed.",
        "Reported model metrics are labeled as recorded artifacts, not independently reproduced or proof of deployment performance.",
        "Backend tests verified: 15 passed, 0 failed, 0 skipped; environment version warning disclosed.",
        "No live sensor integration, CMMS integration, or independent industrial validation is claimed.",
    ]
    for item in checklist:
        story.append(para(f"<font color='#087E67'><b>VERIFIED SCOPE</b></font>  {item}", styles, "small"))
    story.append(Spacer(1, 7))
    story.append(para(
        "Evidence reviewed: project README; backend API, feature pipeline, tests, model metadata, "
        "evaluation reports, saved dataset inventory; frontend implementation; and the prior "
        "project overview PDF. This report describes the repository state checked on "
        "14 November 2026 and should be revised when the implementation or evidence changes.",
        styles,
        "small",
    ))

    document = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=46,
        bottomMargin=44,
        title="AI-Driven Manufacturing Optimization - Project Overview",
        author="Project Documentation",
        subject="Evidence-aligned technical overview of a historical predictive-maintenance prototype",
    )
    document.build(story, canvasmaker=ReportCanvas)
    print(f"Generated: {PDF_OUTPUT}")


if __name__ == "__main__":
    create_document()
