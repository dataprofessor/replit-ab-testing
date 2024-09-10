import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st


def calculate_sample_size(baseline_rate,
                          mde,
                          alpha,
                          power,
                          measurement_type='relative'):
  if measurement_type == 'relative':
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
  else:  # absolute
    p1 = baseline_rate
    p2 = baseline_rate + mde

  p_pooled = (p1 + p2) / 2
  se = np.sqrt(2 * p_pooled * (1 - p_pooled))

  z_alpha = stats.norm.ppf(1 - alpha / 2)
  z_beta = stats.norm.ppf(power)

  n = ((z_alpha + z_beta) / ((p1 - p2) / se))**2
  return int(np.ceil(n))


def plot_sample_size_curve(baseline_rate, mde, alpha, power, measurement_type):
  mde_range = np.linspace(0.01, 0.5, 100)
  sample_sizes = [
      calculate_sample_size(baseline_rate, m, alpha, power, measurement_type)
      for m in mde_range
  ]

  data = pd.DataFrame({'MDE': mde_range, 'Sample Size': sample_sizes})

  line = alt.Chart(data).mark_line().encode(
      x=alt.X('MDE', title='Minimum Detectable Effect'),
      y=alt.Y('Sample Size', scale=alt.Scale(type='log')))

  point = alt.Chart(
      pd.DataFrame({
          'MDE': [mde],
          'Sample Size': [
              calculate_sample_size(baseline_rate, mde, alpha, power,
                                    measurement_type)
          ]
      })).mark_point(color='red', size=100).encode(x='MDE', y='Sample Size')

  chart = (line + point).properties(
      width=600,
      height=400,
      title=
      f'Sample Size vs. MDE (Baseline Rate: {baseline_rate:.0%}, α: {alpha}, Power: {power})'
  )

  return chart


def analyze_results(conversions_a, samples_a, conversions_b, samples_b,
                    confidence_level):
  rate_a = conversions_a / samples_a
  rate_b = conversions_b / samples_b

  se_a = np.sqrt(rate_a * (1 - rate_a) / samples_a)
  se_b = np.sqrt(rate_b * (1 - rate_b) / samples_b)

  z = (rate_b - rate_a) / np.sqrt(se_a**2 + se_b**2)
  p_value = 2 * (1 - stats.norm.cdf(abs(z)))

  ci_a_lower, ci_a_upper = stats.norm.interval(confidence_level,
                                               loc=rate_a,
                                               scale=se_a)
  ci_b_lower, ci_b_upper = stats.norm.interval(confidence_level,
                                               loc=rate_b,
                                               scale=se_b)

  return rate_a, rate_b, p_value, ci_a_lower, ci_a_upper, ci_b_lower, ci_b_upper


st.set_page_config(
    page_title="AB Testing Tools",
    page_icon="assets/microscope.png",
)

col1, col2 = st.columns([4, 2])
with col1:
  st.title("🧪 AB Testing Tools")
with col2:
  measurement_type = "Relative"

  measurement_type = st.radio("Measurement type", ['Relative', 'Absolute'],
                              key="ssc_measurement_type")
  measurement_type = measurement_type.lower()

tab1, tab2 = st.tabs(["Sample Size Calculator", "Results Analyzer"])

with tab1:
  col1, col2 = st.columns([1, 1])

  with col1:
    baseline_conv = st.slider("Baseline conversion rate",
                              0.01,
                              0.50,
                              st.session_state.get('ssc_baseline', 0.10),
                              0.01,
                              key="ssc_baseline",
                              format="%0.2f")

    mde = st.slider(
        f"Minimum Detectable Effect **({measurement_type.lower()})**",
        0.01,
        0.50,
        st.session_state.get('ssc_mde', 0.05),
        0.01,
        key="ssc_mde",
        format="%0.2f")

  with col2:
    alpha = st.slider("Significance level (alpha)",
                      0.01,
                      0.1,
                      st.session_state.get('ssc_alpha', 0.05),
                      0.01,
                      key="ssc_alpha")
    power = st.slider("Statistical power",
                      0.7,
                      0.99,
                      st.session_state.get('ssc_power', 0.8),
                      0.01,
                      key="ssc_power")

  # Visualization of conversion rate range
  if measurement_type == 'relative':
    min_conv = baseline_conv
    max_conv = baseline_conv * (1 + mde)
  else:
    min_conv = baseline_conv
    max_conv = baseline_conv + mde

  sample_size = calculate_sample_size(baseline_conv, mde, alpha, power,
                                      measurement_type)
  st.success(f"Required sample size per group: **{sample_size}**")

  st.altair_chart(plot_sample_size_curve(baseline_conv, mde, alpha, power,
                                         measurement_type),
                  use_container_width=True)

with tab2:

  col1, col2, col3 = st.columns([2, 2, 1])
  with col1:
    conversions_a = st.number_input("Conversions (A)",
                                    0,
                                    None,
                                    st.session_state.get('conv_a', 100),
                                    key="conv_a")
    samples_a = st.number_input("Total samples (A)",
                                1,
                                None,
                                st.session_state.get('samples_a', 1000),
                                key="samples_a")
  with col2:
    conversions_b = st.number_input("Conversions (B)",
                                    0,
                                    None,
                                    st.session_state.get('conv_b', 120),
                                    key="conv_b")
    samples_b = st.number_input("Total samples (B)",
                                1,
                                None,
                                st.session_state.get('samples_b', 1000),
                                key="samples_b")
  with col3:
    confidence_level = st.slider("Confidence Level",
                                 0.8,
                                 0.99,
                                 st.session_state.get('conf_level', 0.95),
                                 0.01,
                                 key="conf_level")
    min_p = round(1 - confidence_level, 2)

  # Analyze results in real-time
  conv_rate_a, conv_rate_b, p_value, ci_a_lower, ci_a_upper, ci_b_lower, ci_b_upper = analyze_results(
      conversions_a, samples_a, conversions_b, samples_b, confidence_level)

  results = pd.DataFrame(
      data={
          "Metric": ["Conversion Rate (A)", "Conversion Rate (B)"],
          "Value": [f"{conv_rate_a:.4f}", f"{conv_rate_b:.4f}"],
          f"{confidence_level*100:.0f}% Confidence Interval": [
              f"[{ci_a_lower:.4f}, {ci_a_upper:.4f}]",
              f"[{ci_b_lower:.4f}, {ci_b_upper:.4f}]"
          ]
      })

  data = pd.DataFrame({
      'Group': ['A', 'B'],
      'Conversion Rate': [conv_rate_a, conv_rate_b],
      'CI Lower': [ci_a_lower, ci_b_lower],
      'CI Upper': [ci_a_upper, ci_b_upper]
  })

  col1, col2 = st.columns([3, 1])
  with col1:
    if p_value < min_p:
      st.success(
          f"P-value: **{p_value:.2f}**. The difference is statistically significant (p < {min_p})"
      )
    else:
      st.warning(
          f"P-value: **{p_value:.2f}**. The difference is not statistically significant (p >= {min_p})"
      )
  with col2:
    if measurement_type == 'relative':
      impact_pct = round(100 * (conv_rate_b - conv_rate_a) / conv_rate_a, 2)
    else:
      impact_pct = round(100 * (conv_rate_b - conv_rate_a), 2)

    st.metric(
        f"**Impact ({measurement_type})**",
        value="",
        delta=f"{impact_pct}%",
    )

  # Chart creation code
  base = alt.Chart(data).encode(x=alt.X('Group', axis=alt.Axis(labelAngle=0)))

  bar = base.mark_bar(opacity=0.8, color='#5276A7').encode(
      y=alt.Y('Conversion Rate',
              scale=alt.Scale(domain=[0, max(ci_a_upper, ci_b_upper) * 1.1])))

  error_bars = base.mark_errorbar(color='#1F3F77').encode(
      y=alt.Y('CI Lower', title='Conversion Rate'), y2=alt.Y2('CI Upper'))

  point = base.mark_point(size=100,
                          color='#1F3F77').encode(y=alt.Y('Conversion Rate'))

  final_chart = (bar + error_bars + point).properties(
      width=600, height=400).configure_axis(
          labelFontSize=12,
          titleFontSize=14,
      ).configure_title(fontSize=16)

  st.altair_chart(final_chart, use_container_width=True)

  col1, col2, col3 = st.columns([.25, .5, .25])
  with col1:
    pass
  with col2:
    st.dataframe(results, hide_index=True)
  with col3:
    pass

# Footer
st.markdown(
    "<div style='text-align: center; color: grey;'>Made with ❤️ by <a href='https://x.com/mattppal'>Matt</a> @ Replit. Source code <a href='https://replit.com/@matt/Streamlit-AB-Testing?v=1'>here</a>.</div>",
    unsafe_allow_html=True)
