from usmle_step1_syllabus import USMLE_STEP1_SYLLABUS


def _leaves(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list):
        yield from node


def test_default_syllabus_is_english_usmle_step1():
    assert list(USMLE_STEP1_SYLLABUS) == ["USMLE Step 1"]
    sections = USMLE_STEP1_SYLLABUS["USMLE Step 1"]

    assert "Foundational Science" in sections
    assert "Organ Systems" in sections
    assert "Biostatistics, Epidemiology, and Ethics" in sections


def test_default_syllabus_is_comprehensive_and_contains_no_chinese_topics():
    leaves = list(_leaves(USMLE_STEP1_SYLLABUS))

    assert len(leaves) >= 100
    assert all(isinstance(topic, str) and topic.strip() for topic in leaves)
    assert not any("\u4e00" <= character <= "\u9fff" for topic in leaves for character in topic)
    assert "Cell injury, adaptation, and death" in leaves
    assert "Antimicrobial pharmacology" in leaves
    assert "Informed consent and decision-making capacity" in leaves

