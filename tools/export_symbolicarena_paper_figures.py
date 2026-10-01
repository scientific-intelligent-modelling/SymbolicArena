import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path
from xml.etree import ElementTree

import requests


SOURCE_URL = "https://arxiv.org/src/2609.35113v1"
SOURCE_SHA256 = "b74b78718fc3387d283f6fe6d00445b9eb6642f3345fd2ad02b214b3de3a7c7e"
WORK_DIR = Path(".agent/work/PAGES-PAPER-002")
OUTPUT_DIR = Path("docs/figures")
FIGURES = {
    "figure-1-overview": "imgs/MainContent/SA-Overview.pdf",
    "figure-2a-coverage": "imgs/Appendix/core50_coverage_notitle.pdf",
    "figure-2b-tradeoff": "imgs/Appendix/core50_tradeoff_notitle.pdf",
    "figure-2c-composition": "imgs/Appendix/core50_composition_full_task.pdf",
    "figure-3a-id": "imgs/Terms20260921/full664_core50_id_1.pdf",
    "figure-3b-ood": "imgs/Terms20260921/full664_core50_ood_1.pdf",
    "figure-4-six-axis": "imgs/Terms20260919/radar15_equal_size.pdf",
    "figure-5-calibration": "imgs/Appendix/durl_probe.pdf",
    "figure-6-probes": "imgs/Appendix/ProbeSelect_explore_ragion.pdf",
    "figure-7-architecture": "imgs/Appendix/SA-pipeline.pdf",
    "figure-8-correlations": "imgs/Appendix/clean_spearman.pdf",
    "figure-9-confidence": "imgs/Final20260917/clean_six_axis_ci.pdf",
}


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = WORK_DIR / "paper-source.tar"
    if not archive_path.is_file():
        response = requests.get(SOURCE_URL, timeout=120)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"arXiv source checksum mismatch: {digest}")

    with tarfile.open(archive_path, "r:gz") as archive:
        for name, member_name in FIGURES.items():
            member = archive.getmember(member_name)
            if not member.isfile():
                raise ValueError(f"Expected a PDF file: {member_name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read PDF file: {member_name}")
            pdf_path = WORK_DIR / f"{name}.pdf"
            with pdf_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            svg_path = OUTPUT_DIR / f"{name}.svg"
            subprocess.run(["pdftocairo", "-svg", str(pdf_path), str(svg_path)], check=True)
            ElementTree.parse(svg_path)
            print(svg_path, svg_path.stat().st_size)


if __name__ == "__main__":
    main()
