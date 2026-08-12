import argparse
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


def compute_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_pass_fail(score: float) -> str:
    return "Pass" if score >= 50 else "Fail"


def generate_recommendations(score: float, attendance: float, assignments: float) -> list[str]:
    recommendations = []

    if score < 60:
        recommendations.append("Focus on core topics and review class notes every day.")
    else:
        recommendations.append("Maintain your study rhythm and reinforce strengths with practice tests.")

    if attendance < 90:
        recommendations.append("Increase attendance to at least 90% to improve exposure to key lessons.")
    else:
        recommendations.append("Continue your strong attendance to stay consistent in learning.")

    if assignments < 8:
        recommendations.append("Complete more assignments and use feedback to improve your understanding.")
    else:
        recommendations.append("Keep submitting assignments on time to strengthen your performance.")

    if score >= 85 and attendance >= 90 and assignments >= 8:
        recommendations.append("You are on track for excellent performance; consider peer tutoring or advanced practice.")

    return recommendations


def build_report_content(student_inputs: dict, predicted_score: float) -> list:
    grade = compute_grade(predicted_score)
    status = compute_pass_fail(predicted_score)
    recommendations = generate_recommendations(predicted_score, student_inputs["attendance_pct"], student_inputs["assignments_completed"])

    title_style = ParagraphStyle(
        name="Title",
        fontSize=20,
        leading=24,
        spaceAfter=12,
        alignment=1,
    )
    heading_style = ParagraphStyle(
        name="Heading",
        fontSize=14,
        leading=18,
        spaceAfter=8,
        textColor=colors.HexColor("#0B5394"),
    )
    body_style = ParagraphStyle(
        name="Body",
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )

    content = [Paragraph("Student Performance Report", title_style), Spacer(1, 12)]

    content.append(Paragraph("Student Inputs", heading_style))
    student_table_data = [["Input", "Value"]]
    for key, value in student_inputs.items():
        student_table_data.append([key.replace("_", " ").title(), str(value)])
    student_table = Table(student_table_data, hAlign="LEFT", colWidths=[2.5 * inch, 3.5 * inch])
    student_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ])
    )
    content.append(student_table)
    content.append(Spacer(1, 14))

    content.append(Paragraph("Prediction Summary", heading_style))
    summary_data = [
        ["Predicted Score", f"{predicted_score:.1f}"],
        ["Grade", grade],
        ["Status", status],
    ]
    summary_table = Table(summary_data, hAlign="LEFT", colWidths=[2.5 * inch, 3.5 * inch])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ])
    )
    content.append(summary_table)
    content.append(Spacer(1, 14))

    content.append(Paragraph("Personalized Recommendations", heading_style))
    for recommendation in recommendations:
        content.append(Paragraph(f"• {recommendation}", body_style))

    return content


def save_pdf_report(output_path: Path, student_inputs: dict, predicted_score: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    content = build_report_content(student_inputs, predicted_score)
    doc.build(content)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a student performance PDF report.")
    parser.add_argument("--hours", type=float, required=True, help="Hours studied")
    parser.add_argument("--attendance", type=float, required=True, help="Attendance percentage")
    parser.add_argument("--previous", type=float, required=True, help="Previous score")
    parser.add_argument("--assignments", type=float, required=True, help="Assignments completed")
    parser.add_argument("--predicted-score", type=float, required=True, help="Predicted score")
    parser.add_argument("--output", default="reports/student_report.pdf", help="Output PDF file path")
    return parser.parse_args()


def main():
    args = parse_args()
    student_inputs = {
        "hours_studied": args.hours,
        "attendance_pct": args.attendance,
        "previous_score": args.previous,
        "assignments_completed": args.assignments,
    }
    output_path = Path(args.output)
    save_pdf_report(output_path, student_inputs, args.predicted_score)
    print(f"Saved student report to {output_path}")


if __name__ == "__main__":
    main()
