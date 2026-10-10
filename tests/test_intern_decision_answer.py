import unittest


class FakeEngine:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return self.response


class InternDecisionAnswerTests(unittest.TestCase):
    def test_intern_answer_uses_static_options_and_same_timestep_image(self):
        from InternDecision_scripts.answer_backend import AnswerBackend

        response = {"answers": {"answer": {"type": "choice", "choice": "white", "confidence": 0.8, "probabilities": {"white": 0.8, "black": 0.2}}}, "model": "fake"}
        engine = FakeEngine(response)
        backend = AnswerBackend(mode="intern_decision", engine=engine, answer_options={"attribute": ["white", "black"]})
        result = backend.answer("What color?", "attribute", {"perception": {}, "images": ["/tmp/frame.png"]})

        self.assertEqual(result.text, "white")
        self.assertEqual(result.status, "ok")
        self.assertEqual(engine.requests[0]["images"], ["/tmp/frame.png"])
        self.assertEqual(engine.requests[0]["questions"]["answer"]["criteria"], {"white": "white", "black": "black"})

    def test_missing_static_options_falls_back_to_qwen(self):
        from InternDecision_scripts.answer_backend import AnswerBackend

        backend = AnswerBackend(mode="intern_decision", engine=FakeEngine({}), answer_options={}, qwen_fn=lambda *args: "qwen answer")
        result = backend.answer("Open?", "existence", {"perception": {}})
        self.assertEqual((result.text, result.status), ("qwen answer", "fallback_qwen"))

    def test_dual_failure_is_serializable_error(self):
        from InternDecision_scripts.answer_backend import AnswerBackend

        backend = AnswerBackend(mode="intern_decision", engine=FakeEngine({}), answer_options={}, qwen_fn=lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
        result = backend.answer("Open?", "existence", {"perception": {}})
        self.assertEqual(result.text, "")
        self.assertEqual(result.status, "error")
        self.assertTrue(result.error_type)


if __name__ == "__main__":
    unittest.main()
