import PyPDF2

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def smart_feedback(prediction):
    suggestions = {
        "Software Engineer": "Add more project experience with backend or system design.",
        "Data Scientist": "Include Kaggle profiles or ML projects with metrics.",
        "Product Manager": "Emphasize leadership and cross-functional collaboration.",
        "UI/UX Designer": "Showcase a portfolio link and design thinking process.",
        "DevOps Engineer": "Mention CI/CD, containerization, or cloud deployment experience."
    }
    return suggestions.get(prediction, "Try to include more technical or project-based content.")
