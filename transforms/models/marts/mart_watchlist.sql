select
    canonical_borrower_id,
    quarter,
    count(distinct lender_cik) as num_lenders,
    avg(mark) as avg_mark,
    min(mark) as worst_mark,
    max(mark_dispersion) as max_dispersion,
    sum(pik_migration) as pik_migrations,
    sum(par_migration) as par_migrations,
    sum(non_accrual_migration) as non_accrual_migrations,
    max(case when non_accrual_flag = 1 then 1 else 0 end) as any_non_accrual,
    avg(mark_drift) as avg_mark_drift,
    max(industry) as industry
from {{ ref('mart_signals') }}
group by canonical_borrower_id, quarter
having
    avg(mark) < 0.85
    or max(mark_dispersion) > 0.10
    or sum(pik_migration) > 0
    or sum(non_accrual_migration) > 0
    or max(case when non_accrual_flag = 1 then 1 else 0 end) = 1
