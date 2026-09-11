"""Meaningful regression checks for isolated proposed controls; no inference."""
import copy
import unittest

from protocol_audit import _response_text, apply_score_policy, parse_counters, parse_detection_response, shuffled_candidates


class ProtocolControlsTest(unittest.TestCase):
    def test_order_is_independent_of_ground_truth_prefix(self):
        # Identical membership with different GT-first arrangements must produce
        # exactly the same permutation. The utility receives no true class.
        a = ["cat", "dog", "bird", "chair"]
        b = ["chair", "bird", "dog", "cat"]
        self.assertEqual(shuffled_candidates(a, seed=91, item_id="image-17"),
                         shuffled_candidates(b, seed=91, item_id="image-17"))
        self.assertEqual(set(a), set(shuffled_candidates(a, seed=91, item_id="image-17")))
        firsts = {shuffled_candidates(a, seed=91, item_id=str(i))[0] for i in range(100)}
        self.assertGreater(len(firsts), 1)

    def test_bare_array_uses_only_retained_prompt_class(self):
        result = parse_detection_response("[1,2,30,40,0.8]", prompt_metadata={
            "single_label": "stapler", "coordinate_mode": "pixel"})
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.detections[0], {"label": "stapler", "box_2d": [1., 2., 30., 40.], "score": .8})

    def test_bare_array_without_class_or_units_never_guesses(self):
        for metadata in (None, {"coordinate_mode": "pixel"}, {"single_label": "cat"}):
            self.assertEqual(parse_detection_response("[1,2,3,4]", prompt_metadata=metadata).status, "failure")

    def test_conflicting_returned_class_is_rejected(self):
        result = parse_detection_response('{"label":"dog","box_2d":[1,2,3,4]}',
                                          prompt_metadata={"single_label": "cat", "coordinate_mode": "pixel"})
        self.assertEqual(result.status, "failure")

    def test_empty_is_not_null_missing_or_malformed(self):
        inputs = ["[]", None, "null", "", "not JSON", '{"box_2d":[1,2,3,4]}',
                  '[{"label":"cat","box_2d":[1,2,3,4]}]']
        results = [parse_detection_response(x) for x in inputs]
        self.assertEqual(parse_counters(results), {"total": 7, "valid_nonempty": 1, "valid_empty": 1, "failure": 5})
        self.assertEqual(results[1].reason, "missing_response")
        self.assertEqual(results[2].reason, "json_null")
        self.assertEqual(results[4].reason, "malformed_json")

    def test_reasoning_only_incomplete_response_has_no_output_text(self):
        raw = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
               "output": [{"type": "reasoning", "content": None}]}
        self.assertIsNone(_response_text(raw))
        self.assertEqual(parse_detection_response(_response_text(raw)).reason, "missing_response")

    def test_invalid_boxes_and_partial_arrays_are_not_silently_accepted(self):
        for text in ('[{"label":"cat","box_2d":[3,2,1,4]}]',
                     '[{"label":"cat","box_2d":[1,2,3,4]},null]',
                     '{"label":"cat","box_2d":[1,2,NaN,4]}'):
            self.assertEqual(parse_detection_response(text).status, "failure")

    def test_native_uniform_preserve_zero_boxes_and_do_not_mutate_input(self):
        detections = [{"label": "cat", "box_2d": [1,2,3,4], "score": 0},
                      {"label": "dog", "box_2d": [1,2,3,4], "confidence": .4},
                      {"label": "bird", "box_2d": [1,2,3,4]}]
        before = copy.deepcopy(detections)
        self.assertEqual([d["score"] for d in apply_score_policy(detections, "native")], [0., .4, 1.])
        self.assertEqual([d["score"] for d in apply_score_policy(detections, "uniform")], [1., 1., 1.])
        self.assertEqual(detections, before)
        zero = parse_detection_response('{"label":"cat","box_2d":[1,2,3,4],"score":0}')
        self.assertEqual(zero.status, "valid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
