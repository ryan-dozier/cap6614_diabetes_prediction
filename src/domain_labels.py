"""Human-readable names for project dataset domains."""

DOMAIN_DISPLAY_NAMES = {
    "source": "Original Synthetic",
    "kaggle": "Playground S5E12",
    "nhanes": "NHANES 2015–2016",
}


def display_domain_name(domain: str, *, include_id: bool = False) -> str:
    name = DOMAIN_DISPLAY_NAMES[domain]
    return f"{name} ({domain})" if include_id else name
