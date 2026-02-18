"""End-to-end integration tests."""

from __future__ import annotations

import numpy as np

from lmm import LMM, Pipeline, DharmaLMM
from lmm.dharma.energy import PrajnaTerm, SilaTerm
from lmm.dharma.engine import UniversalDharmaEngine
from lmm.dharma.interpreter import DharmaInterpreter
from lmm.llm.embeddings import ngram_vectors, cosine_similarity
from lmm.integrations.langchain import DharmaDocumentCompressor, DharmaExampleSelector


class TestEndToEndLMM:
    def test_full_lmm_pipeline(self):
        """Test complete LMM pipeline: data -> surprise -> QUBO -> solve -> result."""
        rng = np.random.RandomState(42)
        data = rng.randn(200)
        model = LMM(k=10, solver_method="greedy")
        model.fit(data)
        result = model.select(data)

        assert len(result.selected_indices) == 10
        assert len(result.surprise_values) == 10
        assert all(0 <= i < 200 for i in result.selected_indices)

    def test_pipeline_orchestrator(self):
        """Test Pipeline with SmartSelector + SmartProcessor."""
        rng = np.random.RandomState(42)
        data = rng.randn(100)
        pipe = Pipeline(k=5)
        pipe.fit(data)
        result = pipe.run(data)

        assert len(result.selection.indices) == 5
        assert hasattr(result, "processing")

    def test_dharma_lmm_full(self):
        """Test DharmaLMM full pipeline."""
        rng = np.random.RandomState(42)
        data = rng.randn(50)

        dlmm = DharmaLMM(k=5)
        dlmm.fit(data)
        result = dlmm.select_dharma(data)
        assert result is not None
        assert hasattr(result, "selected_indices")


class TestEndToEndDharma:
    def test_engine_to_interpreter(self):
        """Test UniversalDharmaEngine -> DharmaInterpreter pipeline."""
        rng = np.random.RandomState(42)
        n = 20
        surprises = rng.rand(n)

        engine = UniversalDharmaEngine(n_variables=n)
        engine.add(PrajnaTerm(surprises))
        engine.add(SilaTerm(k=5, weight=10.0))
        result = engine.synthesize_and_solve(k=5)

        interpreter = DharmaInterpreter()
        interp = interpreter.interpret(result.selected_indices, surprises, k=5)
        assert hasattr(interp, "prajna_score")
        assert hasattr(interp, "narrative")
        assert len(interp.narrative) > 0


class TestEndToEndRAG:
    def test_document_compressor_rerank(self):
        """Test DharmaDocumentCompressor end-to-end."""
        class FakeDoc:
            def __init__(self, text):
                self.page_content = text

        docs = [FakeDoc(f"Document about {topic}")
                for topic in ["quantum", "biology", "quantum", "physics",
                              "chemistry", "quantum", "math", "biology",
                              "physics", "quantum"]]

        compressor = DharmaDocumentCompressor(k=5)
        reranked = compressor.compress_documents(docs, query="quantum physics")
        assert len(reranked) == 5

    def test_example_selector_rerank(self):
        """Test DharmaExampleSelector end-to-end."""
        examples = [
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What color is the sky?", "output": "Blue"},
            {"input": "What is machine learning?", "output": "A branch of AI"},
            {"input": "How does photosynthesis work?", "output": "Plants convert sunlight"},
            {"input": "What is 3*5?", "output": "15"},
            {"input": "Explain gravity", "output": "Attraction between masses"},
        ]
        selector = DharmaExampleSelector(examples=examples, k=3)
        selected = selector.select_examples({"input": "What is 5+3?"})
        assert len(selected) == 3


class TestEndToEndEmbeddings:
    def test_ngram_similarity_pipeline(self):
        """Test ngram vectorization -> cosine similarity."""
        texts = ["quantum computing is amazing",
                 "quantum physics explores particles",
                 "cooking recipes for beginners"]
        vecs = ngram_vectors(texts)
        query_vec = ngram_vectors(["quantum science"])[0]
        sims = cosine_similarity(vecs, query_vec)

        # Quantum texts should be more similar to query than cooking
        assert sims[0] > sims[2]
        assert sims[1] > sims[2]


class TestVersionAndImports:
    def test_version(self):
        import lmm
        assert lmm.__version__ == "1.0.0"

    def test_public_api(self):
        import lmm
        assert hasattr(lmm, "LMM")
        assert hasattr(lmm, "LMMResult")
        assert hasattr(lmm, "QUBOBuilder")
        assert hasattr(lmm, "ClassicalQUBOSolver")
        assert hasattr(lmm, "SubmodularSelector")
        assert hasattr(lmm, "SurpriseCalculator")
        assert hasattr(lmm, "SmartSelector")
        assert hasattr(lmm, "SmartProcessor")
        assert hasattr(lmm, "Pipeline")
        assert hasattr(lmm, "DharmaLMM")
