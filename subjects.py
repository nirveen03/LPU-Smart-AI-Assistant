SUBJECTS = {
    "PEL132": "Communication Skills-II",
    "MTH401": "Discrete Mathematics",
    "INT335": "Design Thinking",
    "CSE306": "Computer Networks",
    "CSE276": "Artificial Intelligence Foundations",
    "CSE205": "Data Structures and Algorithms",
    "CSE202": "Object Oriented Programming",
}

def subject_from_filename(filename: str):
    """Find a known subject code in a PDF filename."""
    upper = filename.upper()
    for code, name in SUBJECTS.items():
        if code in upper:
            return code, name
    return "UNKNOWN", "Unclassified document"
