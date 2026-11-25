"""Export the exact rows and aggregates used by the work log dashboard."""
import csv
from io import BytesIO, StringIO
import re

HEADERS = ["日期", "项目 / 公共事务", "产线 / 工位", "工作阶段", "具体工作内容", "耗时（小时）", "产出物 / 备注"]


def detail_rows(report):
    for r in report["entries"]:
        yield [r["work_date"], r["project_name"] or r["public_affair"], r["scope_name"],
               r["task_category"], r["work_content"], r["duration_minutes"] / 60, r["deliverable_note"]]


def export_report(report, fmt):
    if fmt == "csv":
        output = StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(HEADERS)
        for row in detail_rows(report):
            writer.writerow(["'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@"))
                             else v for v in row])
        return output.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8"
    if fmt != "xlsx":
        raise ValueError("请选择 Excel 或 CSV。")
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    book = Workbook()
    def sheet(ws, title, headings, rows, widths):
        ws.title = title
        ws.append(headings)
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for i, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
        for index, row in enumerate(rows, start=2):
            for column, value in enumerate(row, start=1):
                cell = ws.cell(index, column)
                if isinstance(value, str):
                    cell.value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", value)
                    cell.data_type = "s"
                else:
                    cell.value = value
        ws.auto_filter.ref = ws.dimensions

    sheet(book.active, "工作明细", HEADERS, detail_rows(report), [14, 26, 28, 35, 60, 15, 45])
    for key, title, heading in [("projects", "项目汇总", "项目 / 公共事务"), ("tasks", "任务类型汇总", "任务类型")]:
        ws = book.create_sheet()
        total = report["total_minutes"]
        rows = ((r["name"], r["minutes"] / 60, r["minutes"] / total) for r in report[key])
        sheet(ws, title, [heading, "总工时（小时）", "占比"], rows, [40, 20, 14])
        for cell in list(ws.columns)[2][1:]:
            cell.number_format = "0.0%"
    output = BytesIO()
    book.save(output)
    return output.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
