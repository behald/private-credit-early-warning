with source as (
    select * from {{ source('edgar', 'raw_soi_positions') }}
),

cleaned as (
    select
        cik,
        bdc_name,
        accession_number,
        cast(filing_date as date) as filing_date,
        cast(period_end as date) as period_end,
        fiscal_year,
        fiscal_period,
        frame,
        investment_id,
        borrower_name_raw,
        cast(fair_value as double) as fair_value,
        cast(cost as double) as cost,
        industry,
        investment_type,
        -- Derive quarter from period_end
        date_trunc('quarter', cast(period_end as date)) as quarter
    from source
    where fair_value is not null
        and borrower_name_raw is not null
)

select * from cleaned
