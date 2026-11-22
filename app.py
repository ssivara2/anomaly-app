import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Transaction Monitor', layout='wide')

# ---- load the trained model (made in the Colab notebook, not changed) ----
bundle = joblib.load('model.joblib')
model = bundle['model']
history_scores = bundle['history_scores']
features = bundle['features']

# The columns every transaction file must have, and the number columns we score.
REQUIRED = ['transaction_id', 'amount', 'customer_typical_amount',
            'transactions_last_hour', 'minutes_since_previous', 'account_age_days']
NUMBERS = REQUIRED[1:]

# Drift check: if the share sent to review is more than double the review budget,
# the cutoff no longer fits this data. We wait for 200 scored transactions and
# at least 10 flags before judging, so a few early flags don't cause a false alarm.
MIN_ROWS_FOR_DRIFT_CHECK = 200
MIN_FLAGS_FOR_DRIFT_CHECK = 10


def score(amount, typical, last_hour, gap, age):
    """Turn one transaction into the five features and return its anomaly score."""
    row = pd.DataFrame([{
        'log_amount': np.log1p(amount),
        'amount_ratio': amount / typical,
        'transactions_last_hour': last_hour,
        'log_gap': np.log1p(gap),
        'log_age': np.log1p(age),
    }])[features]
    return -model.score_samples(row)[0]


# ---- PROTECTION 1: check each transaction's data before scoring it ----
def data_problems(r):
    """Return a list of problems with one transaction. An empty list means it is safe to score."""
    problems = []
    for col in NUMBERS:
        if pd.isna(r[col]):
            problems.append(f'{col} is blank or not a number')
    if problems:
        return problems
    if r['amount'] <= 0:
        problems.append('amount is zero or negative (refund?)')
    if r['customer_typical_amount'] <= 0:
        problems.append('typical amount is zero or negative')
    if r['transactions_last_hour'] < 0 or r['minutes_since_previous'] < 0 or r['account_age_days'] < 0:
        problems.append('a count, time gap or account age is negative')
    return problems


# ---- PROTECTION 2: does the share sent to review match the budget? ----
def drift_message(n_flagged, n_scored, pct):
    """Return a warning if the flagged share is far from the review budget, otherwise None."""
    if n_scored < MIN_ROWS_FOR_DRIFT_CHECK or n_flagged < MIN_FLAGS_FOR_DRIFT_CHECK:
        return None
    share = n_flagged / n_scored * 100
    if share > 2 * pct:
        return (f'**These transactions don\'t look like the history the model learned from.** '
                f'{share:.1f}% are being sent to review, but the budget is {pct}%. '
                'The cutoff was set on past transactions, so it no longer fits this data. '
                'Many of these flags are likely normal transactions, and the review team may be overloaded.')
    return None


# ---- sidebar: the review budget ----
st.title('Transaction Monitor')
st.write('Each new transaction is scored by Isolation Forest. '
         'If the score is above the cutoff, it goes to review.')

pct = st.sidebar.slider('Send this share of transactions to review (%)', 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f'Score cutoff: **{cutoff:.3f}**')
st.sidebar.caption('The cutoff is set from past transactions: '
                   'a transaction scoring above it is more unusual than the top '
                   f'{pct}% of history.')

tab1, tab2 = st.tabs(['Check one transaction', 'Live stream'])

# ---- tab 1: type in one transaction ----
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input('Amount ($)', min_value=1.0, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=1.0, value=50.0)
    last_hour = c1.number_input('Transactions in the last hour', min_value=0, value=1)
    gap = c2.number_input('Minutes since previous transaction', min_value=0.0, value=120.0)
    age = c2.number_input('Account age (days)', min_value=0, value=700)

    if st.button('Score this transaction'):
        s = score(amount, typical, last_hour, gap, age)
        if s > cutoff:
            st.error(f'REVIEW  (score {s:.3f}, cutoff {cutoff:.3f})')
        else:
            st.success(f'OK  (score {s:.3f}, cutoff {cutoff:.3f})')
        share = (history_scores < s).mean() * 100
        st.write(f'This is more unusual than {share:.0f}% of past transactions.')

# ---- tab 2: transactions arriving one by one ----
with tab2:
    st.write('Press the button. The app receives the new transactions one at a time '
             'and scores each one as it arrives.')

    # Use the demo file, or upload your own (needed to test the protections).
    uploaded = st.file_uploader('Use your own transactions file (optional). '
                                'If you skip this, the app uses new_transactions.csv.', type='csv')
    speed = st.select_slider('Speed', options=['Slow', 'Medium', 'Fast'], value='Medium')
    pause = {'Slow': 0.5, 'Medium': 0.15, 'Fast': 0.02}[speed]

    if st.button('Start the stream'):
        stream = pd.read_csv(uploaded if uploaded is not None else 'new_transactions.csv')

        # PROTECTION 1a: stop before scoring if a column is missing.
        missing = [c for c in REQUIRED if c not in stream.columns]
        if missing:
            st.error(f'**The stream was not started.** This file is missing these columns: '
                     f'{", ".join(missing)}. No transactions were scored or approved. '
                     f'Fix the file so it has all of these columns: {", ".join(REQUIRED)}, '
                     'then upload it again.')
            st.stop()

        # Text like "n/a" becomes blank, so it is caught by the row check below.
        for col in NUMBERS:
            stream[col] = pd.to_numeric(stream[col], errors='coerce')

        box = st.empty()
        counts = st.empty()
        drift_box = st.empty()
        st.subheader('Sent to review')
        table = st.empty()
        st.subheader('Could not be scored: check by hand')
        bad_table = st.empty()
        flagged = []
        not_scored = []

        for i, r in stream.iterrows():
            # PROTECTION 1b: a row with bad data is never scored and never shown as ok.
            problems = data_problems(r)
            if problems:
                not_scored.append({'id': r['transaction_id'], 'amount': r['amount'],
                                   'problem': '; '.join(problems)})
                box.warning(f"{r['transaction_id']}: NOT SCORED -> {'; '.join(problems)}")
                bad_table.dataframe(pd.DataFrame(not_scored), hide_index=True)
            else:
                s = score(r['amount'], r['customer_typical_amount'],
                          r['transactions_last_hour'], r['minutes_since_previous'],
                          r['account_age_days'])
                msg = f"{r['transaction_id']}: ${r['amount']:.2f}  score {s:.3f}"
                if s > cutoff:
                    flagged.append({'id': r['transaction_id'], 'amount': r['amount'],
                                    'typical': r['customer_typical_amount'],
                                    'last_hour': r['transactions_last_hour'],
                                    'score': round(s, 3)})
                    box.error(msg + '  ->  REVIEW')
                else:
                    box.info(msg + '  ->  ok')

            n_scored = i + 1 - len(not_scored)
            counts.write(f'Received **{i + 1}** of {len(stream)}.  '
                         f'Sent to review: **{len(flagged)}**.  '
                         f'Could not be scored: **{len(not_scored)}**')
            if flagged:
                # Highest score first, so reviewers start with the most unusual ones.
                table.dataframe(pd.DataFrame(flagged).sort_values('score', ascending=False),
                                hide_index=True)

            # PROTECTION 2: warn as soon as the review share drifts far from the budget.
            warning = drift_message(len(flagged), n_scored, pct)
            if warning:
                drift_box.warning(warning)
            else:
                drift_box.empty()
            time.sleep(pause)

        drift_box.empty()   # the final summary below repeats it with what to do
        if not not_scored:
            bad_table.write('None. Every transaction had valid data.')

        # ---- the final summary: only a plain green "Done" if nothing went wrong ----
        n_scored = len(stream) - len(not_scored)
        warning = drift_message(len(flagged), n_scored, pct)

        if not_scored:
            st.warning(f'**{len(not_scored)} transactions could not be scored** because of bad data '
                       '(see the table above). They were NOT approved. '
                       'Hold these payments and check them by hand, and ask the data team '
                       'to fix the source file.')
        if warning:
            budget = max(1, round(n_scored * pct / 100))
            st.warning(warning + f'\n\n**What to do:** if your team can only review about {budget} '
                       f'transactions (your {pct}% budget), start at the top of the review table, '
                       'which is sorted by score. Don\'t just move the slider to make the warning '
                       'go away. Tell the analytics team the model needs to be checked '
                       'against recent transactions.')
        if not not_scored and not warning:
            st.success(f'Done. {len(flagged)} transactions were sent to review.')
        else:
            st.info(f'Finished with problems. {len(flagged)} sent to review, '
                    f'{len(not_scored)} could not be scored.')

        # if the file has an answer key, check how we did
        if 'needs_review' in stream.columns:
            answer = stream.set_index('transaction_id')['needs_review']
            hits = sum(answer[f['id']] for f in flagged)
            st.write(f'{hits} of the {len(flagged)} flagged transactions really needed '
                     f'review. In total, {answer.sum()} transactions needed review.')
