select
    canonical_borrower_id,
    max(industry) as industry,
    min(quarter) as first_seen_quarter,
    max(quarter) as last_seen_quarter,
    count(distinct quarter) as quarters_observed,
    count(distinct lender_cik) as total_lenders,
    avg(mark) as lifetime_avg_mark,
    min(mark) as lifetime_min_mark,
    max(mark_dispersion) as max_cross_lender_dispersion,
    max(case when non_accrual_flag = 1 then 1 else 0 end) as ever_non_accrual,
    max(case when pik_flag = 1 then 1 else 0 end) as ever_pik,
    sum(fair_value) as latest_total_fv
from {{ ref('mart_signals') }}
group by canonical_borrower_id
