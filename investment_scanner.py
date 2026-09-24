"""Analyst-first stock research scanner. Offline by default; no order execution."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scanner_data as data


@dataclass(frozen=True)
class Settings:
    # Heuristics to validate prospectively, not fitted or proven trading thresholds.
    min_ratings: int = 10
    min_buy_pct: float = 70
    max_sell_pct: float = 10
    min_upside_pct: float = 15
    min_turnover: float = 20_000_000
    max_price_age_days: float = 4  # calendar days, allowing weekends
    max_analyst_age_days: float = 7  # retrieval age, not analyst publication age
    min_week_drop_pct: float = 3
    max_week_drop_pct: float = 10
    min_fortnight_drop_pct: float = 5
    max_fortnight_drop_pct: float = 15
    max_drop_sigma: float = 3
    max_target_spread_pct: float = 60
    max_below_200d_pct: float = 10
    max_month_drop_pct: float = 20
    max_eps_cut_pct: float = 5
    earnings_buffer_days: float = 7
    upside_cap_pct: float = 50

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Settings must be finite, nonnegative numbers')
        if self.min_ratings < 1 or int(self.min_ratings) != self.min_ratings:
            raise ValueError('min_ratings must be a positive integer')
        if not 50 <= self.min_buy_pct <= 100 or self.max_sell_pct > 100:
            raise ValueError('Use majority buy support and percentages within 0..100')
        if self.min_upside_pct >= self.upside_cap_pct:
            raise ValueError('upside_cap_pct must exceed min_upside_pct')
        if self.min_week_drop_pct >= self.max_week_drop_pct or self.min_fortnight_drop_pct >= self.max_fortnight_drop_pct:
            raise ValueError('Pullback minimum must be less than maximum')


def now_utc(value=None):
    stamp = pd.Timestamp(value if value is not None else datetime.now(timezone.utc))
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def num(row, name):
    return data.safe_float(row.get(name))


def age_days(value, now):
    stamp = pd.to_datetime(value, utc=True, errors='coerce')
    return (now - stamp).total_seconds() / 86400 if pd.notna(stamp) else np.nan


def price_metrics(symbol, history, as_of=None):
    """Use completed cached daily observations; 5 sessions need 6 closes."""
    if not isinstance(history, pd.DataFrame) or history.empty:
        return {'symbol': symbol, 'price_error': 'Missing price history'}
    h = history.copy().sort_index()
    h = h[~h.index.duplicated(keep='last')]
    market_now = now_utc(as_of).tz_convert('America/New_York')
    # Yahoo can include today's still-forming bar. Exclude it before US close.
    if market_now.hour < 16:
        h = h[[pd.Timestamp(index).date() != market_now.date() for index in h.index]]
    if h.empty:
        return {'symbol': symbol, 'price_error': 'No completed daily observations'}
    column = next((c for c in ['Close', 'Adj Close', 'close'] if c in h), None)
    if column is None:
        return {'symbol': symbol, 'price_error': 'Missing close column'}
    close = pd.to_numeric(h[column], errors='coerce')
    if (~np.isfinite(close) | (close <= 0)).any():
        return {'symbol': symbol, 'price_error': 'Invalid or missing daily closes; refresh history'}
    h['Close'] = close
    row = data.calculate_price_metrics(symbol, h)
    # Averages require their full stated observation window.
    for window in [50, 200]:
        row[f'moving_average_{window}d'] = close.tail(window).mean() if len(close) >= window else np.nan
        row[f'distance_to_{window}d_ma_pct'] = 100 * (close.iloc[-1] / row[f'moving_average_{window}d'] - 1)
    volume = pd.to_numeric(h.get('Volume', pd.Series(index=h.index, dtype=float)), errors='coerce')
    dollar_volume = (close * volume.where(volume >= 0)).tail(20)
    row['average_daily_turnover_20d'] = dollar_volume.mean() if dollar_volume.count() == 20 else np.nan
    changes = close.pct_change(fill_method=None).dropna()
    row.update(
        history_sessions=len(close),
        return_1d_pct=100 * data.series_return(close, 1),
        return_5d_pct=100 * data.series_return(close, 5),
        return_10d_pct=100 * data.series_return(close, 10),
        drawdown_20d_pct=100 * (close.iloc[-1] / close.tail(20).max() - 1) if len(close) >= 20 else np.nan,
        daily_volatility_pct=100 * changes.tail(20).std() if len(changes) >= 20 else np.nan,
        above_5d_average=bool(len(close) >= 5 and close.iloc[-1] >= close.tail(5).mean()),
    )
    return row


def evaluate(row, settings, now):
    s = settings
    flags, blocks, review = [], [], []
    def block(reason):
        blocks.append(reason)
    price = num(row, 'price_close')  # One consistent dated price for targets and pullbacks.
    price_age = age_days(row.get('price_as_of'), now)
    analyst_age = age_days(row.get('fetched_at_utc'), now)
    if not np.isfinite(price) or price <= 0:
        block('MISSING_VALID_PRICE')
    if not np.isfinite(price_age) or price_age < 0 or price_age > s.max_price_age_days:
        block('PRICE_REFRESH_REQUIRED')
    if not np.isfinite(analyst_age) or analyst_age < 0 or analyst_age > s.max_analyst_age_days:
        block('ANALYST_REFRESH_REQUIRED')
    counts = [num(row, k) for k in ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']]
    valid_counts = all(np.isfinite(v) and v >= 0 and v.is_integer() for v in counts)
    total = sum(counts) if valid_counts else np.nan
    valid = row.get('ratings_valid') is True or isinstance(row.get('ratings_valid'), np.bool_) and bool(row['ratings_valid'])
    if not valid or not valid_counts or not total > 0:
        block('MISSING_CURRENT_RATINGS')
        positive = negative = strong = np.nan
    else:
        strong = counts[0] / total * 100
        positive = (counts[0] + counts[1]) / total * 100
        negative = (counts[3] + counts[4]) / total * 100
    if not total >= s.min_ratings:
        block('INSUFFICIENT_RATING_COVERAGE')
    if not positive >= s.min_buy_pct:
        block('BUY_SUPPORT_BELOW_MINIMUM')
    if not negative <= s.max_sell_pct:
        block('SELL_SUPPORT_TOO_HIGH')
    median, mean = num(row, 'target_median'), num(row, 'target_mean')
    target = median if median > 0 else mean if mean > 0 else np.nan
    target_source = 'Median' if median > 0 else 'Mean fallback' if mean > 0 else 'Unavailable'
    upside = 100 * (target / price - 1) if price > 0 else np.nan
    if not np.isfinite(upside) or upside + 1e-9 < s.min_upside_pct:
        block('UPSIDE_BELOW_MINIMUM_OR_MISSING')
    if not num(row, 'average_daily_turnover_20d') >= s.min_turnover:
        block('LOW_OR_UNKNOWN_LIQUIDITY')
    if target_source == 'Mean fallback':
        flags.append('MEAN_TARGET_FALLBACK')
    low, high = num(row, 'target_low'), num(row, 'target_high')
    spread = 100 * (high - low) / target if target > 0 and high >= low > 0 else np.nan
    if (low > 0 and target < low) or (high > 0 and target > high) or (high > 0 and low > high):
        block('INCONSISTENT_TARGET_RANGE')
    if not np.isfinite(spread):
        review.append('TARGET_SPREAD_UNKNOWN')
    elif spread > s.max_target_spread_pct:
        review.append('WIDE_TARGET_DISAGREEMENT')
    # Fetch time is deliberately not presented as the date the analysts updated their reports.
    flags.append('ANALYST_PUBLICATION_DATES_UNKNOWN')
    if str(row.get('component_errors', '') or '').strip():
        flags.append('PARTIAL_PROVIDER_DATA')
    r5, r10 = num(row, 'return_5d_pct'), num(row, 'return_10d_pct')
    week = -s.max_week_drop_pct <= r5 <= -s.min_week_drop_pct
    fortnight = -s.max_fortnight_drop_pct <= r10 <= -s.min_fortnight_drop_pct and r5 < 0
    pullback = bool(week or fortnight)
    days = 5 if week else 10
    drop = -r5 if week else -r10
    daily_vol = num(row, 'daily_volatility_pct')
    sigma = drop / (daily_vol * np.sqrt(days)) if daily_vol > 0 and pullback else np.nan
    if not np.isfinite(daily_vol) or daily_vol <= 0:
        review.append('VOLATILITY_UNKNOWN')
    if r5 < -s.max_week_drop_pct or r10 < -s.max_fortnight_drop_pct or (pullback and sigma > s.max_drop_sigma):
        review.append('EXTREME_DECLINE')
    trend = num(row, 'distance_to_200d_ma_pct')
    if not np.isfinite(trend):
        review.append('LONG_TERM_TREND_UNKNOWN')
    elif trend < -s.max_below_200d_pct or num(row, 'return_1m_pct') < -s.max_month_drop_pct:
        review.append('DAMAGED_LONG_TERM_TREND')
    eps = num(row, 'eps_estimate_change_30d_pct')
    if not np.isfinite(eps):
        review.append('EPS_REVISION_DATA_UNKNOWN')
    elif eps < -s.max_eps_cut_pct:
        review.append('FALLING_EARNINGS_ESTIMATES')
    earnings_age = age_days(row.get('next_earnings_date_utc'), now)
    days_to_earnings = -earnings_age
    if not np.isfinite(days_to_earnings) or days_to_earnings < 0:
        review.append('NEXT_EARNINGS_DATE_UNKNOWN')
    elif days_to_earnings <= s.earnings_buffer_days:
        review.append('EARNINGS_SOON')
    stabilizing = bool(num(row, 'return_1d_pct') > 0 and row.get('above_5d_average') == True)
    thesis = not blocks
    if blocks:
        status = 'DATA_REFRESH_REQUIRED' if any('REFRESH' in b for b in blocks) else 'DOES_NOT_QUALIFY'
    elif review:
        status = 'REVIEW_RISK'
    elif not pullback:
        status = 'WAIT_FOR_PULLBACK'
    elif not stabilizing:
        status = 'WAIT_FOR_STABILIZATION'
    else:
        status = 'PULLBACK_CANDIDATE'
    # Consensus bands ensure large upside cannot outrank a stronger conviction band.
    tier = 3 if positive >= 90 else 2 if positive >= 80 else 1 if positive >= s.min_buy_pct else 0
    conviction = (0.8 * positive + 0.2 * strong) if np.isfinite(positive) else np.nan
    upside_points = np.clip(upside, 0, s.upside_cap_pct) / s.upside_cap_pct * 100
    # Moderate dips receive a bounded preference; crashes never earn unlimited points.
    dip_points = 100 * max(0, 1 - abs(sigma - 1.5) / 1.5) if pullback and np.isfinite(sigma) else 0
    score = 0.60 * conviction + 0.25 * upside_points + 0.15 * dip_points
    score = float(score) if np.isfinite(score) else np.nan
    reasons = blocks + review
    if not reasons:
        reasons = ['Strong analyst support and upside; ' + {
            'PULLBACK_CANDIDATE': 'moderate pullback with initial stabilization',
            'WAIT_FOR_PULLBACK': 'wait for a qualifying recent decline',
            'WAIT_FOR_STABILIZATION': 'pullback present, recovery confirmation absent',
        }[status]]
    return dict(current_price=price, price_source='Latest cached adjusted daily close',
                price_age_days=price_age, analyst_retrieval_age_days=analyst_age,
                rating_count=total, positive_rating_pct=positive, strong_buy_pct=strong,
                negative_rating_pct=negative, selected_target=target, target_source=target_source,
                target_upside_pct=upside, target_spread_pct=spread,
                target_low_upside_pct=100 * (low / price - 1) if low > 0 and price > 0 else np.nan,
                conviction_tier=tier, analyst_conviction_score=conviction,
                upside_score=upside_points, pullback_score=dip_points, research_score=score,
                has_pullback=pullback, pullback_sigma=sigma, stabilizing=stabilizing,
                thesis_eligible=thesis, status=status, days_to_earnings=days_to_earnings,
                risk_flags=' | '.join(dict.fromkeys(blocks + review + flags)),
                explanation='; '.join(reasons), scan_as_of_utc=now.isoformat())


def rank_stocks(frame, settings=None, as_of=None):
    settings, now = settings or Settings(), now_utc(as_of)
    if frame.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS + ['thesis_eligible'])
    if 'symbol' not in frame or frame['symbol'].duplicated().any():
        raise ValueError('One unique row per symbol is required')
    metrics = pd.DataFrame([evaluate(row, settings, now) for _, row in frame.iterrows()], index=frame.index)
    ranked = pd.concat([frame.drop(columns=metrics.columns, errors='ignore'), metrics], axis=1)
    ranked = ranked.sort_values(['thesis_eligible', 'conviction_tier', 'research_score',
                                 'positive_rating_pct', 'strong_buy_pct', 'symbol'],
                                ascending=[False, False, False, False, False, True], na_position='last')
    ranked = ranked.reset_index(drop=True)
    ranked['research_rank'] = np.arange(1, len(ranked) + 1)
    return ranked


DISPLAY_COLUMNS = ['research_rank', 'symbol', 'company_name', 'status', 'positive_rating_pct',
                   'strong_buy_pct', 'rating_count', 'current_price', 'selected_target',
                   'target_upside_pct', 'return_5d_pct', 'return_10d_pct',
                   'research_score', 'explanation']


def scan_cache(base_dir, settings=None, as_of=None):
    base = Path(base_dir)
    cache = base / 'cache'
    universe_path = cache / 'index_constituents.csv'
    if not universe_path.exists():  # Backward compatibility with existing S&P-only caches.
        universe_path = cache / 'sp500_constituents.csv'
    universe = pd.read_csv(universe_path)
    rows = []
    for _, constituent in universe.iterrows():
        symbol = str(constituent['symbol'])
        row = constituent.to_dict()
        try:
            row.update(price_metrics(symbol, pd.read_pickle(cache / 'price_history_2y' / f'{symbol}.pkl'), as_of))
        except Exception as exc:
            row.update(price_error=f'Price cache could not be read: {exc}')
        try:
            bundle = pd.read_pickle(cache / 'fundamentals' / f'{symbol}.pkl')
            if bundle.get('symbol') != symbol:
                raise ValueError('Cache symbol mismatch')
            row.update(data.normalize_saved_yahoo_bundle(bundle))
        except Exception as exc:
            row.update(component_errors=f'Analyst cache could not be read: {exc}')
        row['company_name'] = row.get('company_name_yahoo') or row.get('security') or symbol
        rows.append(row)
    return rank_stocks(pd.DataFrame(rows), settings, as_of)


def save_results(ranked, output_dir, settings=None):
    """Keep the screen, exports, and methodology on the same ranking."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        'ranked': ranked,
        'analyst_watchlist': ranked.loc[ranked['thesis_eligible'].eq(True)],
        'pullback_candidates': ranked.loc[ranked['status'].eq('PULLBACK_CANDIDATE')],
        'review_or_wait': ranked.loc[ranked['thesis_eligible'].eq(True) & ~ranked['status'].eq('PULLBACK_CANDIDATE')],
    }
    paths = {}
    for name, frame in frames.items():
        path = output / f'{name}_latest.csv'
        columns = [c for c in DISPLAY_COLUMNS if c in frame]
        frame[columns + [c for c in frame if c not in columns]].to_csv(path, index=False)
        paths[name] = path
    index_counts = {}
    if 'index_name' in ranked:
        for memberships in ranked['index_name'].fillna('').astype(str):
            for membership in (name.strip() for name in memberships.split('|')):
                if membership:
                    index_counts[membership] = index_counts.get(membership, 0) + 1
    manifest = {
        'generated_at_utc': now_utc().isoformat(),
        'scan_as_of_utc': ranked['scan_as_of_utc'].iloc[0] if len(ranked) else None,
        'settings': asdict(settings or Settings()),
        'status_counts': {str(k): int(v) for k, v in ranked['status'].value_counts().items()},
        'index_counts': index_counts,
        'methodology': 'Analyst conviction bands, then 60% conviction + 25% capped target upside + 15% bounded pullback score. Read README.md for gates.',
        'limitations': 'Research heuristic, not a return probability. No recovery date inferred. Retrieval age is not publication age. Historical analysis of current snapshots is not a backtest.',
    }
    (output / 'scan_manifest.json').write_text(json.dumps(manifest, indent=2))
    return paths


def refresh_cache(base_dir, settings=None, symbols=None, indexes=data.SUPPORTED_INDEXES):
    """Explicit, incremental network refresh; retain old data when a request fails."""
    s, base = settings or Settings(), Path(base_dir)
    cache = base / 'cache'
    for folder in ['price_history_2y', 'fundamentals', 'snapshots']:
        (cache / folder).mkdir(parents=True, exist_ok=True)
    data.yf.set_tz_cache_location(str(cache / 'yfinance_timezone_cache'))
    constituent_path = cache / 'index_constituents.csv'
    refresh_universe = not constituent_path.exists()
    if not refresh_universe:
        refresh_universe = (time.time() - constituent_path.stat().st_mtime) / 86400 > 7
    if refresh_universe:
        universe = data.fetch_index_constituents(cache, indexes)
        universe.to_csv(constituent_path, index=False)
    else:
        universe = pd.read_csv(constituent_path)
    wanted = set(symbols) if symbols else set(universe['symbol'])
    unknown = wanted - set(universe['symbol'])
    if unknown:
        raise ValueError(f'Symbols outside cached universe: {sorted(unknown)}')
    now, errors = now_utc(), []
    for index, symbol in enumerate(sorted(wanted), 1):
        print(f'[{index}/{len(wanted)}] {symbol}', flush=True)
        price_path = cache / 'price_history_2y' / f'{symbol}.pkl'
        try:
            history = pd.read_pickle(price_path)
            age = age_days(history.index[-1], now)
        except Exception:
            age = np.nan
        # Daily refresh for prices even when still within the shortlist tolerance.
        if not np.isfinite(age) or age < 0 or age > 1:
            response = data.call_with_retry(lambda: data.download_batch_history([symbol], '2y', 30), f'{symbol} prices', max_attempts=2)
            history = data.extract_ticker_frame(response.value, symbol) if isinstance(response.value, pd.DataFrame) else pd.DataFrame()
            if not history.empty and not price_metrics(symbol, history).get('price_error'):
                temp = price_path.with_suffix('.tmp')
                history.to_pickle(temp)
                temp.replace(price_path)
            else:
                errors.append({'symbol': symbol, 'error': response.error or 'Invalid price history'})
        bundle_path = cache / 'fundamentals' / f'{symbol}.pkl'
        try:
            previous = pd.read_pickle(bundle_path)
            age = age_days(previous.get('fetched_at_utc'), now)
            successful = previous.get('fetch_success', False)
        except Exception:
            age, successful = np.nan, False
        if not np.isfinite(age) or age < 0 or age > s.max_analyst_age_days or not successful:
            ticker = data.yf.Ticker(symbol)
            bundle = {'symbol': symbol, 'fetched_at_utc': now_utc().isoformat()}
            for name, getter in [('info', ticker.get_info), ('analyst_price_targets', ticker.get_analyst_price_targets),
                                 ('recommendations', ticker.get_recommendations), ('earnings_estimate', ticker.get_earnings_estimate),
                                 ('eps_trend', ticker.get_eps_trend), ('eps_revisions', ticker.get_eps_revisions)]:
                response = data.call_with_retry(getter, f'{symbol} {name}', max_attempts=2)
                bundle[name] = data.component_to_dict(response)
            normalized = data.normalize_saved_yahoo_bundle(bundle)
            target_ok = any(num(normalized, k) > 0 for k in ['target_median', 'target_mean'])
            bundle['fetch_success'] = bool(normalized['ratings_valid'] and target_ok)
            snapshot = cache / 'snapshots' / f'{symbol}_{now_utc().strftime("%Y%m%dT%H%M%S%f")}.pkl'
            pd.to_pickle(bundle, snapshot)
            if bundle['fetch_success']:
                if bundle_path.exists():
                    old_snapshot = cache / 'snapshots' / f'{symbol}_previous_{now_utc().strftime("%Y%m%dT%H%M%S%f")}.pkl'
                    old_snapshot.write_bytes(bundle_path.read_bytes())
                temp = bundle_path.with_suffix('.tmp')
                pd.to_pickle(bundle, temp)
                temp.replace(bundle_path)
            else:
                errors.append({'symbol': symbol, 'error': 'Incomplete current targets/ratings; previous bundle retained'})
        time.sleep(0.3)
    pd.DataFrame(errors, columns=['symbol', 'error']).to_csv(cache / 'refresh_errors.csv', index=False)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'sp500_notebook_data')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--refresh', action='store_true', help='Fetch stale/missing Yahoo data before analysis')
    parser.add_argument('--symbols', nargs='+', help='Limit network refresh to these symbols')
    parser.add_argument('--indexes', nargs='+', choices=data.SUPPORTED_INDEXES,
                        default=list(data.SUPPORTED_INDEXES),
                        help='Indexes to refresh (default: S&P 500 and STOXX Europe 600)')
    args = parser.parse_args()
    settings = Settings()
    if args.refresh:
        refresh_cache(args.data_dir, settings, args.symbols, args.indexes)
    ranked = scan_cache(args.data_dir, settings)
    paths = save_results(ranked, args.output_dir or args.data_dir / 'output', settings)
    print(ranked['status'].value_counts().to_string())
    print(f'Pullback candidates: {ranked.status.eq("PULLBACK_CANDIDATE").sum()}')
    for label, path in paths.items():
        print(f'{label}: {path}')


if __name__ == '__main__':
    main()
