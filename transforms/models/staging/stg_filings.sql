with source as (
    select * from {{ source('edgar', 'raw_filings') }}
),

cleaned as (
    select
        cik,
        bdc_name,
        form,
        accession_number,
        cast(filing_date as date) as filing_date,
        primary_document
    from source
)

select * from cleaned
