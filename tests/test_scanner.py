import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import investment_scanner as scanner
import scanner_data as data

NOW = '2026-09-09T20:00:00Z'


def stock(**overrides):
    row = dict(symbol='TEST', company_name='Test company', price_close=100,
               price_as_of='2026-09-09', fetched_at_utc='2026-09-08',
               strong_buy=10, buy=8, hold=2, sell=0, strong_sell=0, ratings_valid=True,
               target_median=125, target_mean=130, target_low=110, target_high=150,
               average_daily_turnover_20d=100_000_000, return_5d_pct=-5,
               return_10d_pct=-6, daily_volatility_pct=2, return_1d_pct=1,
               above_5d_average=True, distance_to_200d_ma_pct=5, return_1m_pct=-5,
               eps_estimate_change_30d_pct=1, next_earnings_date_utc='2026-10-25')
    row.update(overrides)
    return row


def evaluate(**overrides):
    return scanner.rank_stocks(pd.DataFrame([stock(**overrides)]), as_of=NOW).iloc[0]


class ScannerTests(unittest.TestCase):
    def test_good_pullback(self):
        row = evaluate()
        self.assertEqual(row.status, 'PULLBACK_CANDIDATE')
        self.assertEqual(row.target_upside_pct, 25)
        self.assertEqual(row.positive_rating_pct, 90)
        self.assertEqual(row.rating_count, 20)

    def test_falling_price_is_not_recovery_confirmation(self):
        self.assertEqual(evaluate(return_1d_pct=-2).status, 'WAIT_FOR_STABILIZATION')
        self.assertEqual(evaluate(above_5d_average=False).status, 'WAIT_FOR_STABILIZATION')

    def test_no_decline_waits(self):
        self.assertEqual(evaluate(return_5d_pct=5, return_10d_pct=10).status, 'WAIT_FOR_PULLBACK')

    def test_large_upside_cannot_override_consensus_band(self):
        a = stock(symbol='HIGH_CONVICTION', target_median=120)
        b = stock(symbol='HUGE_TARGET', strong_buy=8, buy=8, hold=4,
                  target_median=400, target_high=500)
        ranked = scanner.rank_stocks(pd.DataFrame([b, a]), as_of=NOW)
        self.assertEqual(ranked.symbol.iloc[0], 'HIGH_CONVICTION')

    def test_strong_buy_and_upside_improve_rank_within_band(self):
        rows = [stock(symbol='LESS_STRONG', strong_buy=5, buy=13),
                stock(symbol='MORE_STRONG'), stock(symbol='MORE_UPSIDE', target_median=140)]
        ranked = scanner.rank_stocks(pd.DataFrame(rows), as_of=NOW)
        self.assertEqual(list(ranked.symbol), ['MORE_UPSIDE', 'MORE_STRONG', 'LESS_STRONG'])

    def test_earnings_analysts_cannot_replace_rating_coverage(self):
        row = evaluate(strong_buy=2, buy=2, hold=0, earnings_estimate_analysts=100)
        self.assertFalse(row.thesis_eligible)
        self.assertEqual(row.rating_count, 4)

    def test_missing_partial_negative_fractional_counts_rejected(self):
        for change in [dict(hold=np.nan), dict(sell=-1), dict(buy=8.2), dict(ratings_valid=False)]:
            with self.subTest(change=change):
                self.assertFalse(evaluate(**change).thesis_eligible)

    def test_zero_and_missing_price_do_not_create_upside(self):
        for price in [0, -1, np.nan, np.inf]:
            self.assertFalse(evaluate(price_close=price).thesis_eligible)

    def test_missing_and_inconsistent_targets(self):
        for change in [dict(target_median=np.nan, target_mean=np.nan), dict(target_low=160), dict(target_high=90)]:
            self.assertFalse(evaluate(**change).thesis_eligible)
        row = evaluate(target_median=0)
        self.assertEqual(row.selected_target, 130)
        self.assertEqual(row.target_source, 'Mean fallback')

    def test_stale_and_future_data_are_blocked(self):
        for change in [dict(price_as_of='2026-07-28'), dict(fetched_at_utc='2026-07-27'),
                       dict(price_as_of='2026-09-10'), dict(fetched_at_utc='2026-09-10'),
                       dict(price_as_of=None)]:
            self.assertEqual(evaluate(**change).status, 'DATA_REFRESH_REQUIRED')

    def test_risks_cannot_qualify_as_candidates(self):
        for change in [dict(return_5d_pct=-25), dict(target_high=220),
                       dict(distance_to_200d_ma_pct=-20), dict(return_1m_pct=-25),
                       dict(eps_estimate_change_30d_pct=-10), dict(eps_estimate_change_30d_pct=np.nan),
                       dict(next_earnings_date_utc='2026-09-12'), dict(next_earnings_date_utc='2026-08-01'),
                       dict(next_earnings_date_utc=None), dict(daily_volatility_pct=np.nan),
                       dict(daily_volatility_pct=.1)]:
            with self.subTest(change=change):
                self.assertEqual(evaluate(**change).status, 'REVIEW_RISK')

    def test_threshold_edges_and_two_week_dip(self):
        self.assertEqual(evaluate(return_5d_pct=-3).status, 'PULLBACK_CANDIDATE')
        self.assertEqual(evaluate(return_5d_pct=-1, return_10d_pct=-8).status, 'PULLBACK_CANDIDATE')
        self.assertEqual(evaluate(return_5d_pct=1, return_10d_pct=-8).status, 'WAIT_FOR_PULLBACK')
        self.assertFalse(evaluate(target_median=114.99).thesis_eligible)

    def test_current_recommendation_selection(self):
        frame = pd.DataFrame([dict(period='-1m', strongBuy=100, buy=0, hold=0, sell=0, strongSell=0),
                              dict(period='0m', strongBuy=10, buy=8, hold=2, sell=0, strongSell=0)])
        self.assertEqual(data.recommendation_metrics(frame)['recommendation_total'], 20)
        self.assertFalse(data.recommendation_metrics(frame.iloc[:1])['ratings_valid'])
        self.assertFalse(data.recommendation_metrics(frame.drop(columns='sell'))['ratings_valid'])

    def test_price_returns_use_trading_sessions_not_calendar_days(self):
        h = pd.DataFrame({'Close': [100, 99, 98, 96, 94, 95], 'Volume': 1_000_000},
                         index=pd.bdate_range('2026-08-31', periods=6))
        row = scanner.price_metrics('TEST', h)
        self.assertAlmostEqual(row['return_5d_pct'], -5)
        self.assertTrue(np.isnan(row['return_10d_pct']))
        self.assertTrue(np.isnan(row['moving_average_200d']))
        h.loc[h.index[-1], 'Close'] = np.nan
        self.assertTrue(scanner.price_metrics('TEST', h)['price_error'])

    def test_negative_eps_changes_keep_economic_direction(self):
        worse = pd.DataFrame([{'current': -1.5, '30daysAgo': -1.0}], index=['0q'])
        better = pd.DataFrame([{'current': -.5, '30daysAgo': -1.0}], index=['0q'])
        self.assertEqual(data.eps_metrics(None, worse, None)['eps_estimate_change_30d_pct'], -50)
        self.assertEqual(data.eps_metrics(None, better, None)['eps_estimate_change_30d_pct'], 50)

    def test_in_progress_daily_bar_excluded(self):
        history = pd.DataFrame({'Close': [100, 101, 102], 'Volume': 1_000_000},
                               index=pd.bdate_range('2026-09-07', periods=3))
        before = scanner.price_metrics('TEST', history, '2026-09-09T18:00Z')
        after = scanner.price_metrics('TEST', history, '2026-09-09T21:00Z')
        self.assertEqual(before['price_close'], 101)
        self.assertEqual(after['price_close'], 102)

    def test_upside_minimum_handles_float_rounding(self):
        self.assertTrue(evaluate(target_median=115).thesis_eligible)

    def test_failed_refresh_preserves_original_cache_and_date(self):
        with TemporaryDirectory() as folder:
            base = Path(folder)
            (base/'cache/fundamentals').mkdir(parents=True)
            pd.DataFrame([{'symbol': 'TEST', 'index_name': 'S&P 500'}]).to_csv(base/'cache/index_constituents.csv', index=False)
            path = base/'cache/fundamentals/TEST.pkl'
            pd.to_pickle({'symbol': 'TEST', 'fetched_at_utc': '2020-01-01', 'fetch_success': True}, path)
            before = path.read_bytes()
            with patch.object(data, 'call_with_retry', return_value=data.ApiResult(None, 'Unavailable')), patch.object(data.yf, 'Ticker'), patch.object(data.yf, 'set_tz_cache_location'), patch.object(scanner.time, 'sleep'):
                errors = scanner.refresh_cache(base, symbols=['TEST'])
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(errors)
            self.assertTrue((base/'cache/refresh_errors.csv').exists())
            self.assertTrue(list((base/'cache/snapshots').glob('TEST_*.pkl')))

    def test_duplicates_and_settings_validation(self):
        with self.assertRaises(ValueError):
            scanner.rank_stocks(pd.DataFrame([stock(), stock()]), as_of=NOW)
        with self.assertRaises(ValueError):
            scanner.Settings(min_week_drop_pct=12)

    def test_exports_match_screen_and_include_empty_shortlist(self):
        with TemporaryDirectory() as folder:
            ranked = scanner.rank_stocks(pd.DataFrame([stock(price_as_of='2026-07-28')]), as_of=NOW)
            paths = scanner.save_results(ranked, folder)
            self.assertEqual(len(pd.read_csv(paths['pullback_candidates'])), 0)
            exported = pd.read_csv(paths['ranked'])
            self.assertEqual(exported.status.iloc[0], ranked.status.iloc[0])
            empty = scanner.rank_stocks(pd.DataFrame(), as_of=NOW)
            scanner.save_results(empty, folder)

    def test_first_constituent_download_has_configured_url(self):
        html = '<table><tr><th>Symbol</th><th>Security</th></tr><tr><td>BRK.B</td><td>Berkshire</td></tr></table>'
        with TemporaryDirectory() as folder, patch.object(data.requests, 'get') as get:
            get.return_value.text = html
            path = Path(folder) / 'sp500_constituents.csv'
            frame = data.fetch_sp500_constituents(path)
            self.assertEqual(frame.symbol.iloc[0], 'BRK-B')
            self.assertTrue(path.exists())
            self.assertEqual(get.call_args.args[0], data.SP500_URL)

    def test_stoxx_download_maps_local_tickers_to_yahoo_symbols(self):
        html = '''<table><tr><th>Company</th><th>Ticker</th><th>Country</th><th>Industry</th></tr>
        <tr><td>SAP</td><td>SAP</td><td>Germany</td><td>Software</td></tr>
        <tr><td>AstraZeneca</td><td>AZN</td><td>United Kingdom</td><td>Health Care</td></tr></table>'''
        with TemporaryDirectory() as folder, patch.object(data.requests, 'get') as get:
            get.return_value.text = html
            path = Path(folder) / 'stoxx600_constituents.csv'
            frame = data.fetch_stoxx600_constituents(path)
            self.assertEqual(frame.symbol.tolist(), ['SAP.DE', 'AZN.L'])
            self.assertTrue(frame.index_name.eq('STOXX Europe 600').all())
            self.assertEqual(get.call_args.args[0], data.STOXX600_URL)

    def test_combined_index_universe_deduplicates_and_preserves_memberships(self):
        sp = pd.DataFrame([{'symbol': 'DUAL', 'index_name': 'S&P 500'}])
        eu = pd.DataFrame([
            {'symbol': 'DUAL', 'index_name': 'STOXX Europe 600'},
            {'symbol': 'EU.DE', 'index_name': 'STOXX Europe 600'},
        ])
        with TemporaryDirectory() as folder, \
                patch.object(data, 'fetch_sp500_constituents', return_value=sp), \
                patch.object(data, 'fetch_stoxx600_constituents', return_value=eu):
            combined = data.fetch_index_constituents(Path(folder))
        self.assertEqual(combined.symbol.tolist(), ['DUAL', 'EU.DE'])
        self.assertEqual(combined.iloc[0].index_name, 'S&P 500 | STOXX Europe 600')

    def test_offline_scan_makes_no_yahoo_requests(self):
        with TemporaryDirectory() as folder:
            base = Path(folder)
            (base/'cache').mkdir()
            pd.DataFrame([{'symbol': 'TEST'}]).to_csv(base/'cache/sp500_constituents.csv', index=False)
            with patch.object(data.yf, 'Ticker', side_effect=AssertionError('network')), patch.object(data.yf, 'download', side_effect=AssertionError('network')):
                ranked = scanner.scan_cache(base, as_of=NOW)
            self.assertEqual(len(ranked), 1)
            self.assertFalse(ranked.thesis_eligible.iloc[0])


if __name__ == '__main__':
    unittest.main()
