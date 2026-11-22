import json
import tempfile
import unittest
from pathlib import Path

from eval_pipeline.benchmark import evaluate, grade, numeric_matches, retry_unavailable


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_numeric_units_and_rounding(self):
        self.assertTrue(numeric_matches('Apple reported $416.161 billion [S1].', 416161000000)[0])
        self.assertTrue(numeric_matches('About 51.3% of sales.', 51.3)[0])
        self.assertFalse(numeric_matches('About 15.1% of sales.', 51.3)[0])
        self.assertTrue(numeric_matches(
            'The region reported $60 billion; dividing by total revenue gives 51.3% [S1][S2].',
            51.3, 'What percentage of total revenue came from the region?')[0])

    def test_numeric_match_must_not_belong_to_another_year_or_metric(self):
        question = 'What was Apple revenue in fiscal year 2025?'
        self.assertFalse(numeric_matches('FY2024 revenue was $416.161 billion; '
                                         'FY2025 revenue was $391 billion.',
                                         416161000000, question)[0])
        self.assertFalse(numeric_matches('FY2025 net income was $416.161 billion. '
                                         'FY2025 revenue was $391 billion.',
                                         416161000000, question)[0])

    def test_entity_requires_correct_company_and_number(self):
        question = {'evaluation': 'exact_match_entity', 'gold_answer': 'Alphabet (GOOGL) at 15.1%',
                    'gold_answer_numeric': 15.1}
        self.assertEqual(grade(question, {'answer': 'Alphabet grew 15.1%.'})['status'], 'correct')
        self.assertEqual(grade(question, {'answer': 'Microsoft grew 15.1%.'})['status'], 'incorrect')

    def test_llm_judge_uses_question_expected_answer(self):
        question = {'evaluation': 'llm_judge', 'gold_answer': 'Expected detail'}
        called = []

        def judge(record, expected):
            called.append(expected)
            return {'status': 'partial', 'findings': ['Missing detail'], 'model': 'test'}

        self.assertEqual(grade(question, {'answer': 'Some detail'}, judge)['status'], 'partial')
        self.assertEqual(called, ['Expected detail'])

    def test_independent_benchmark_judge_can_grade_an_unconfirmed_agent_review(self):
        question = {'evaluation': 'llm_judge', 'gold_answer': 'Expected detail'}
        record = {'answer': 'Supported detail [S1].', 'answer_status': 'needs_verification'}
        self.assertEqual(grade(question, record,
                               lambda row, expected: {'status': 'correct'})['status'], 'correct')
        self.assertEqual(grade(question, {**record, 'answer_status': 'calculation_failed'},
                               lambda row, expected: {'status': 'correct'})['status'], 'unavailable')

    def test_evaluation_saves_separate_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'questions').mkdir()
            (root / 'evaluation').mkdir()
            (root / 'questions/dev_questions_with_answers.json').write_text(json.dumps([{
                'id': 'q_001', 'question': 'Revenue?', 'evaluation': 'fuzzy_numeric',
                'gold_answer': '$416.2 billion', 'gold_answer_numeric': 416161000000}]))
            (root / 'evaluation/dev_run.json').write_text(json.dumps({
                'run_id': 'sample', 'questions': [{'id': 'q_001', 'answer': '$416.161 billion'}]}))
            progress = []
            result = evaluate(root, progress=lambda done, total, question_id:
                              progress.append((done, total, question_id)))
            self.assertEqual(result['counts']['correct'], 1)
            self.assertTrue((root / 'evaluation/benchmark_run.json').is_file())
            self.assertEqual(progress, [(0, 1, 'q_001'), (1, 1, None)])

    def test_evaluation_can_grade_an_isolated_answer_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'questions').mkdir()
            (root / 'evaluation').mkdir()
            (root / 'questions/dev_questions_with_answers.json').write_text(json.dumps([{
                'id': 'q_001', 'question': 'Revenue?', 'evaluation': 'fuzzy_numeric',
                'gold_answer': '$416.2 billion', 'gold_answer_numeric': 416161000000}]))
            (root / 'evaluation/dev_run.json').write_text(json.dumps({
                'run_id': 'submitted', 'questions': [{'id': 'q_001', 'answer': 'Original'}]}))
            isolated = root / 'evaluation/verified_comparison'
            isolated.mkdir()
            (isolated / 'dev_run.json').write_text(json.dumps({
                'run_id': 'verified', 'questions': [{'id': 'q_001', 'answer': '$416.161 billion'}]}))
            result = evaluate(root, answer_run_path=isolated / 'dev_run.json',
                              result_path=isolated / 'benchmark_run.json')
            self.assertEqual(result['counts']['correct'], 1)
            self.assertEqual(result['answer_run_id'], 'verified')
            self.assertFalse((root / 'evaluation/benchmark_run.json').exists())

    def test_retry_only_unavailable_judgment_preserves_other_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'questions').mkdir()
            (root / 'evaluation').mkdir()
            questions = [{'id': 'q_006', 'question': 'Risk?', 'evaluation': 'llm_judge',
                          'gold_answer': 'Supply risk'}]
            (root / 'questions/dev_questions_with_answers.json').write_text(json.dumps(questions))
            (root / 'evaluation/dev_run.json').write_text(json.dumps({
                'run_id': 'sample', 'questions': [{'id': 'q_006', 'answer': 'Supply risk [S1]',
                                                  'answer_status': 'needs_verification'}]}))
            (root / 'evaluation/benchmark_run.json').write_text(json.dumps({
                'answer_run_id': 'sample', 'rows': [{'id': 'q_006', 'status': 'unavailable',
                    'answer': 'Supply risk [S1]'}], 'counts': {'unavailable': 1}}))
            called = []

            def judge(record, expected):
                called.append((record['answer'], expected))
                return {'status': 'correct', 'findings': [], 'model': 'test'}

            self.assertEqual(retry_unavailable('q_006', root, judge)['status'], 'correct')
            saved = json.loads((root / 'evaluation/benchmark_run.json').read_text())
            self.assertEqual(saved['counts']['correct'], 1)
            self.assertEqual(saved['counts']['unavailable'], 0)
            self.assertEqual(called, [('Supply risk [S1]', 'Supply risk')])
            with self.assertRaises(ValueError):
                retry_unavailable('q_006', root, judge)


if __name__ == '__main__':
    unittest.main()
