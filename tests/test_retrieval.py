from pathlib import Path
from retrieval import load_passages,Retriever
def test_retrieval():
    x=Retriever(load_passages(Path(__file__).resolve().parents[1]/"docs"))
    hits=x.search("How long are support chat transcripts retained?")
    assert hits and any("90 days" in p.text for p,_ in hits)
def test_abstain():
    x=Retriever(load_passages(Path(__file__).resolve().parents[1]/"docs"))
    assert not x.search("quasars astrophysics orbit redshift",min_score=.1)
