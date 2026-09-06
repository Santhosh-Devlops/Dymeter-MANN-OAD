PAPER_RESULTS = [
    {"dataset": "INSECTS_Abr", "model": "DyMETER", "aucRoc": 0.832, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "INSECTS_Incr", "model": "DyMETER", "aucRoc": 0.810, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "INSECTS_IncrGrd", "model": "DyMETER", "aucRoc": 0.866, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "INSECTS_IncrRecr", "model": "DyMETER", "aucRoc": 0.911, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "INSECTS_Abr", "model": "METER", "aucRoc": 0.824, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "INSECTS_Incr", "model": "METER", "aucRoc": 0.778, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "INSECTS_IncrGrd", "model": "METER", "aucRoc": 0.842, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "INSECTS_IncrRecr", "model": "METER", "aucRoc": 0.908, "aucPr": None, "source": "Stored Reference", "table": "Reference V"},
    {"dataset": "Ion", "model": "DyMETER", "aucRoc": 0.972, "aucPr": None, "source": "Stored Reference", "table": "Reference VIII"},
    {"dataset": "M.T.", "model": "DyMETER", "aucRoc": 0.866, "aucPr": None, "source": "Stored Reference", "table": "Reference VIII"},
    {"dataset": "BGL", "model": "DyMETER", "aucRoc": 0.906, "aucPr": None, "source": "Stored Reference", "table": "Reference VIII"},
]


SUPPORTED_COMPARISON_MODELS = [
    "DyMETER",
    "METER",
    "D3R",
    "SARAD",
    "EnhancedDymeterMANN",
]


def comparisonRows(datasetName: str, localRows: list[dict]) -> list[dict]:
    paperRows = [row for row in PAPER_RESULTS if row["dataset"].lower() == datasetName.lower()]
    return paperRows + localRows
