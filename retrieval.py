from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
@dataclass
class Passage:
    doc:str
    section:str
    text:str
def load_passages(directory):
    result=[]
    for path in sorted(Path(directory).glob("*.md")):
        section="";lines=[]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if section and lines:result.append(Passage(path.name,section," ".join(lines)))
                section=line[3:].strip();lines=[]
            elif section and line.strip():lines.append(line.strip())
        if section and lines:result.append(Passage(path.name,section," ".join(lines)))
    return result
class Retriever:
    def __init__(self,passages):
        if not passages:raise ValueError("No passages")
        self.passages=passages
        self.vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
        self.matrix=self.vectorizer.fit_transform([p.section+" "+p.text for p in passages])
    def search(self,query,min_score=.1,top_k=3):
        if not query.strip():return []
        x=self.vectorizer.transform([query[:600]])
        if x.nnz==0:return []
        scores=cosine_similarity(x,self.matrix)[0]
        return [(self.passages[i],round(float(scores[i]),3)) for i in scores.argsort()[::-1][:top_k] if scores[i]>=min_score]
