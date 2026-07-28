import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "paper" / "neurips26-upload" / "dataset-metadata"
READY_PATH = METADATA_DIR / "croissant.OPENREVIEW_READY.json"
TEMPLATE_PATH = METADATA_DIR / "croissant.TEMPLATE_NEEDS_URL.json"
DIRECT_READY_PATH = (
    ROOT
    / "paper"
    / "neurips26-upload"
    / "upload-ready"
    / "openreview-direct"
    / "croissant.OPENREVIEW_READY.json"
)
UPLOAD_READY_README = (
    ROOT / "paper" / "neurips26-upload" / "upload-ready" / "README.md"
)
PACKAGE_README = ROOT / "paper" / "neurips26-upload" / "README_upload_package.md"

REQUIRED_RAI_FIELDS = {
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
}
LEGACY_INVALID_RAI_FIELDS = {
    "rai:limitations",
    "rai:useCases",
    "rai:maintenancePlan",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return "TODO_REPLACE" in value
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    return False


def _assert_responsible_ai_and_provenance(metadata: dict) -> None:
    assert REQUIRED_RAI_FIELDS <= metadata.keys()
    assert metadata["rai:hasSyntheticData"] is True
    assert isinstance(metadata["rai:dataSocialImpact"], str)

    derived_from = metadata["prov:wasDerivedFrom"]
    assert len(derived_from) >= 4
    assert all(
        isinstance(source, dict)
        and source.get("@type") == "prov:Entity"
        and str(source.get("@id", "")).startswith("https://")
        for source in derived_from
    )

    generated_by = metadata["prov:wasGeneratedBy"]
    assert isinstance(generated_by, dict)
    assert generated_by.get("@type") == "prov:Activity"
    assert generated_by.get("prov:label")

    for field in REQUIRED_RAI_FIELDS | {
        "prov:wasDerivedFrom",
        "prov:wasGeneratedBy",
    }:
        assert not _contains_placeholder(metadata[field]), field


def _assert_minimal_croissant_structure(metadata: dict) -> None:
    context = metadata["@context"]
    assert context["sc"] == "https://schema.org/"
    assert context["dct"] == "http://purl.org/dc/terms/"
    assert context["prov"] == "http://www.w3.org/ns/prov#"
    assert context["conformsTo"] == "dct:conformsTo"
    assert metadata["@type"] == "sc:Dataset"
    assert metadata["conformsTo"] == "http://mlcommons.org/croissant/1.1"

    distributions = metadata["distribution"]
    archive_ids = {
        item["@id"]
        for item in distributions
        if item.get("@type") == "cr:FileObject"
    }
    assert archive_ids
    for item in distributions:
        assert item.get("@id")
        assert item.get("encodingFormat")
        if item.get("@type") == "cr:FileObject":
            assert item.get("contentUrl")
        elif item.get("@type") == "cr:FileSet":
            assert item.get("includes")
            assert item.get("containedIn", {}).get("@id") in archive_ids
        else:
            raise AssertionError(f"Unsupported distribution type: {item.get('@type')}")

    for record_set in metadata["recordSet"]:
        assert record_set.get("@id")
        assert record_set.get("field")
        for field in record_set["field"]:
            assert field.get("@id")
            assert field.get("dataType")
            assert field.get("source") or "value" in field


def test_ready_croissant_has_complete_rai_provenance_and_core_structure() -> None:
    metadata = _load(READY_PATH)
    _assert_responsible_ai_and_provenance(metadata)
    _assert_minimal_croissant_structure(metadata)
    assert not _contains_placeholder(metadata)


def test_template_keeps_only_host_specific_placeholders() -> None:
    metadata = _load(TEMPLATE_PATH)
    _assert_responsible_ai_and_provenance(metadata)
    _assert_minimal_croissant_structure(metadata)


def test_openreview_ready_copy_matches_canonical_metadata() -> None:
    assert _load(DIRECT_READY_PATH) == _load(READY_PATH)


def test_ready_croissant_does_not_use_legacy_invalid_rai_field_names() -> None:
    metadata = _load(READY_PATH)
    assert LEGACY_INVALID_RAI_FIELDS.isdisjoint(metadata.keys())


def test_readmes_point_to_ready_croissant_file() -> None:
    upload_ready_text = UPLOAD_READY_README.read_text(encoding="utf-8")
    package_text = PACKAGE_README.read_text(encoding="utf-8")

    assert "croissant.OPENREVIEW_READY.json" in upload_ready_text
    assert "croissant.OPENREVIEW_NEEDS_URL.json" not in upload_ready_text
    assert "dataset-metadata/croissant.OPENREVIEW_READY.json" in package_text
