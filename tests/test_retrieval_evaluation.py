"""Check official RAGAS retrieval wiring without calling Fireworks."""

import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from devui.retrieval_eval import cases, evaluate, latest_result, retry_unavailable, retrieved_hits
from eval_pipeline.retrieval import _attribute_claim_chunks, _precision_judgments, _recall_claims, build_metrics, score_methods


class FakeMetric:
    def __init__(self, value, fail=False):
        self.value = value
        self.fail = fail
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError('judge timed out')
        return SimpleNamespace(value=self.value)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_server_marks_saved_retrieval_run_complete(self):
        from devui import server
        with patch('devui.retrieval_eval.evaluate', return_value={'question_count': 2, 'run_id': 'pilot'}):
            server.run_retrieval_eval(2)
        self.assertEqual(server.EVAL_JOB['state'], 'complete')
        self.assertEqual(server.EVAL_JOB['done'], 2)
        self.assertEqual(server.EVAL_RESULT['run_id'], 'pilot')

    def test_pilot_uses_cited_pdf_and_sql_from_saved_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'questions').mkdir()
            (root / 'evaluation').mkdir()
            questions = [
                {'id': 'q_006', 'question': 'Supply risk?', 'gold_answer': 'Risk', 'required_modalities': ['pdf']},
                {'id': 'q_012', 'question': 'China sales?', 'gold_answer': 'Sales', 'required_modalities': ['sql', 'pdf']},
                {'id': 'q_017', 'question': 'Other?', 'gold_answer': 'Other', 'required_modalities': ['pdf']},
            ]
            (root / 'questions/dev_questions_with_answers.json').write_text(json.dumps(questions))
            records = [
                {'id': 'q_006', 'answer': 'Risk [S1].', 'sources': [
                    {'id': 'S1', 'type': 'pdf', 'chunk_id': 'pdf-1', 'filename': 'a.pdf', 'pages': [7], 'text': 'Risk'}]},
                {'id': 'q_012', 'answer': 'Sales [S1][S2].', 'sources': [
                    {'id': 'S1', 'type': 'sql', 'sql': 'SELECT revenue', 'rows': [{'revenue': 4}]},
                    {'id': 'S2', 'type': 'pdf', 'chunk_id': 'pdf-2', 'filename': 'a.pdf', 'pages': [8], 'text': 'Sales'}]},
            ]
            (root / 'evaluation/dev_run.json').write_text(json.dumps({'run_id': 'answer-run', 'questions': records}))
            precision, recall = FakeMetric(1), FakeMetric(0.5)
            result = evaluate(root=root, metrics=(precision, recall), question_count=2)
            self.assertEqual(result['question_count'], 2)
            self.assertEqual(result['answer_run_id'], 'answer-run')
            self.assertEqual([hit['type'] for hit in result['rows'][1]['hits']['cited']], ['sql', 'pdf'])
            self.assertIn('revenue', precision.calls[1]['retrieved_contexts'][0])
            self.assertEqual(result['summary']['cited']['recall_scored'], 2)
            self.assertEqual(latest_result(root, 'answer-run')['run_id'], result['run_id'])
            self.assertIsNone(latest_result(root, 'another-run'))

    def test_retry_rescores_only_missing_metric_for_one_saved_question(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'questions').mkdir()
            (root / 'evaluation').mkdir()
            questions = [
                {'id': question_id, 'question': 'Risk?', 'gold_answer': 'Risk',
                 'required_modalities': ['pdf']}
                for question_id in ('q_006', 'q_012')]
            (root / 'questions/dev_questions_with_answers.json').write_text(json.dumps(questions))
            records = [{'id': question_id, 'answer': 'Risk [S1].', 'sources': [
                {'id': 'S1', 'type': 'pdf', 'text': 'Risk', 'filename': 'a.pdf', 'pages': [7]}]}
                for question_id in ('q_006', 'q_012')]
            (root / 'evaluation/dev_run.json').write_text(json.dumps({
                'run_id': 'answer-run', 'questions': records}))
            original = evaluate(root=root, metrics=(FakeMetric(0, fail=True), FakeMetric(1)),
                                question_count=2)
            precision, recall = FakeMetric(0.75), FakeMetric(0.25)
            updated = retry_unavailable('q_006', root=root, metrics=(precision, recall))
            self.assertEqual(len(precision.calls), 1)
            self.assertEqual(recall.calls, [])
            self.assertEqual(updated['rows'][0]['metrics']['cited']['context_precision'], 0.75)
            self.assertEqual(updated['rows'][0]['metrics']['cited']['context_recall'], 1)
            self.assertIsNone(updated['rows'][1]['metrics']['cited']['context_precision'])
            self.assertEqual(updated['summary']['cited']['precision_scored'], 1)
            self.assertEqual(updated['run_id'], original['run_id'])
            self.assertEqual(latest_result(root, 'answer-run')['summary'], updated['summary'])
            with self.assertRaisesRegex(ValueError, 'no unavailable'):
                retry_unavailable('q_006', root=root, metrics=(precision, recall))

    def test_builds_official_ragas_classes(self):
        from ragas.metrics.collections import ContextPrecision, ContextRecall
        with patch.dict('os.environ', {'FIREWORKS_API_KEY': 'test-key'}):
            precision, recall = build_metrics()
        self.assertIsInstance(precision, ContextPrecision)
        self.assertIsInstance(recall, ContextRecall)
        self.assertNotIn("exactly '-'", precision.prompt.instruction)

    def test_exposes_existing_ragas_chunk_verdicts_without_new_judge_calls(self):
        from ragas.metrics.collections.context_precision.util import (
            ContextPrecisionInput, ContextPrecisionOutput, ContextPrecisionPrompt,
        )
        question, reference = 'Where?', 'Paris.'
        contexts = ['Paris is in France.', 'Berlin is in Germany.']
        prompt = ContextPrecisionPrompt()
        cache = {}
        for context, verdict in zip(contexts, (1, 0)):
            key = (ContextPrecisionOutput.__name__, prompt.to_string(ContextPrecisionInput(
                question=question, context=context, answer=reference)))
            cache[key] = SimpleNamespace(verdict=verdict, reason='RAGAS reason')
        scorer = SimpleNamespace(prompt=prompt, llm=SimpleNamespace(cache=cache))
        self.assertEqual(_precision_judgments(scorer, question, reference, contexts), [
            {'relevant': True, 'reason': 'RAGAS reason'},
            {'relevant': False, 'reason': 'RAGAS reason'},
        ])

    def test_exposes_existing_ragas_recall_claims_without_new_judge_calls(self):
        from ragas.metrics.collections.context_recall.util import (
            ContextRecallInput, ContextRecallOutput, ContextRecallPrompt,
        )
        prompt = ContextRecallPrompt()
        key = (ContextRecallOutput.__name__, prompt.to_string(ContextRecallInput(
            question='Where?', context='Paris is in France.', answer='Paris is in France.')))
        output = SimpleNamespace(classifications=[SimpleNamespace(
            statement='Paris is in France.', attributed=1, reason='Directly supported')])
        scorer = SimpleNamespace(prompt=prompt, llm=SimpleNamespace(cache={key: output}))
        self.assertEqual(_recall_claims(scorer, 'Where?', 'Paris is in France.',
                                        ['Paris is in France.']), [
            {'statement': 'Paris is in France.', 'attributed': True,
             'reason': 'Directly supported'}])

    def test_records_only_returned_chunk_ids_for_supported_claims(self):
        class Judge:
            async def agenerate(self, prompt, response_model):
                self.prompt = prompt
                return response_model(links=[{'claim_index': 0, 'chunk_ids': ['chunk-b']}])

        judge = Judge()
        scorer = SimpleNamespace(llm=judge)
        claims = [{'statement': 'Paris is in France.', 'attributed': True, 'reason': 'Yes'},
                  {'statement': 'Berlin is in France.', 'attributed': False, 'reason': 'No'}]
        hits = [{'id': 'chunk-a', 'text': 'Other text'},
                {'id': 'chunk-b', 'text': 'Paris is in France.'}]
        linked = asyncio.run(_attribute_claim_chunks(scorer, claims, hits))
        self.assertEqual(linked[0]['supporting_chunk_ids'], ['chunk-b'])
        self.assertEqual(linked[1]['supporting_chunk_ids'], [])
        self.assertIn('chunk-b', judge.prompt)

        class BadJudge:
            async def agenerate(self, prompt, response_model):
                return response_model(links=[{'claim_index': 0, 'chunk_ids': ['not-returned']}])

        with self.assertRaisesRegex(ValueError, 'unknown chunk ID'):
            asyncio.run(_attribute_claim_chunks(SimpleNamespace(llm=BadJudge()), claims, hits))

    def test_scores_each_ranking_with_question_reference_and_contexts(self):
        precision, recall = FakeMetric(0.5), FakeMetric(0.75)
        methods = {'lexical': [{'text': 'first'}, {'text': 'second'}],
                   'semantic': [{'text': 'second'}, {'text': 'first'}]}
        results = score_methods('What changed?', 'Revenue grew.', methods,
                                metrics=(precision, recall))
        self.assertEqual(results['lexical']['context_precision'], 0.5)
        self.assertEqual(results['semantic']['context_recall'], 0.75)
        self.assertEqual(len(precision.calls), 2)
        self.assertEqual(precision.calls[0], {'user_input': 'What changed?',
                                              'reference': 'Revenue grew.',
                                              'retrieved_contexts': ['first', 'second']})
        self.assertEqual(recall.calls[1]['retrieved_contexts'], ['second', 'first'])

    def test_metric_failure_does_not_replace_other_metric(self):
        with self.assertLogs('eval_pipeline.retrieval', level='WARNING') as logs:
            results = score_methods('Question', 'Answer', {'hybrid': [{'text': 'Evidence'}]},
                                    metrics=(FakeMetric(0, fail=True), FakeMetric(1)),
                                    question_id='q_017')
        self.assertIsNone(results['hybrid']['context_precision'])
        self.assertEqual(results['hybrid']['context_recall'], 1)
        self.assertEqual(results['hybrid']['errors']['context_precision'], 'TimeoutError')
        self.assertIn('question_id=q_017', logs.output[0])
        self.assertIn('metric=context_precision', logs.output[0])
        self.assertIn('detail=judge timed out', logs.output[0])

    def test_cited_hits_excludes_uncited_sources_and_deduplicates_citations(self):
        from devui.retrieval_eval import cited_hits
        record = {'answer': 'One [S2], again [S2].', 'sources': [
            {'id': 'S1', 'type': 'pdf', 'chunk_id': 'unused', 'text': 'Unused'},
            {'id': 'S2', 'type': 'pdf', 'chunk_id': 'used', 'text': 'Useful',
             'filename': 'f.pdf', 'pages': [5]}]}
        self.assertEqual([hit['id'] for hit in cited_hits(record)], ['used'])

    def test_retrieved_hits_include_uncited_sources_in_tool_exposure_order(self):
        record = {'answer': 'Answer [S3].', 'sources': [
            {'id': 'S1', 'type': 'pdf', 'chunk_id': 'first', 'text': 'First passage'},
            {'id': 'S2', 'type': 'pdf', 'chunk_id': 'second', 'text': 'Second passage'},
            {'id': 'S3', 'type': 'sql', 'sql': 'SELECT revenue', 'rows': [{'revenue': 4}]},
        ], 'trace': [
            {'tool': 'search_filings', 'source_ids': ['S2', 'S1']},
            {'tool': 'search_filings', 'source_ids': ['S1']},
            {'tool': 'query_financials', 'source_ids': ['S3']},
        ]}
        self.assertEqual([hit['id'] for hit in retrieved_hits(record)],
                         ['second', 'first', 'S3'])

    def test_scores_cited_and_all_exposed_evidence_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'questions').mkdir()
            (root / 'evaluation').mkdir()
            (root / 'questions/dev_questions_with_answers.json').write_text(json.dumps([
                {'id': question_id, 'question': 'What changed?', 'gold_answer': 'Revenue rose.',
                 'required_modalities': ['pdf']} for question_id in ('q_006', 'q_012')]))
            records = [{'id': question_id, 'answer': 'Revenue rose [S2].', 'sources': [
                {'id': 'S1', 'type': 'pdf', 'chunk_id': 'irrelevant', 'text': 'Table of contents'},
                {'id': 'S2', 'type': 'pdf', 'chunk_id': 'relevant', 'text': 'Revenue rose'}],
                'trace': [{'tool': 'search_filings', 'source_ids': ['S1', 'S2']}]
            } for question_id in ('q_006', 'q_012')]
            (root / 'evaluation/dev_run.json').write_text(json.dumps({
                'run_id': 'saved-run', 'questions': records}))
            precision, recall = FakeMetric(0.75), FakeMetric(1)
            result = evaluate(root=root, metrics=(precision, recall), question_count=2)
            self.assertEqual([hit['id'] for hit in result['rows'][0]['hits']['cited']], ['relevant'])
            self.assertEqual([hit['id'] for hit in result['rows'][0]['hits']['retrieved']],
                             ['irrelevant', 'relevant'])
            self.assertEqual(precision.calls[0]['retrieved_contexts'], ['Revenue rose'])
            self.assertEqual(precision.calls[1]['retrieved_contexts'],
                             ['Table of contents', 'Revenue rose'])
            self.assertEqual(result['summary']['retrieved']['precision_scored'], 2)

    def test_cited_sql_evidence_explains_revenue_units(self):
        from devui.retrieval_eval import cited_hits
        record = {'answer': 'Revenue was $416.161 billion [S1].', 'sources': [
            {'id': 'S1', 'type': 'sql', 'sql': 'SELECT revenue',
             'rows': [{'company_ticker': 'AAPL', 'fiscal_year': 2025,
                       'revenue': 416161000000}]}]}
        self.assertIn('revenue $416.161 billion USD', cited_hits(record)[0]['text'])


if __name__ == '__main__':
    unittest.main()
