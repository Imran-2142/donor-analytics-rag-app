import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Import the enhanced RAG system
from enhanced_rag_system_old_v1 import EnhancedDonorRAGSystem

# Page configuration
st.set_page_config(
    page_title="Enhanced Donor Analytics RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .enhanced-badge {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 1rem;
    }
    .query-type-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .analytics-container {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_enhanced_rag_system():
    """Initialize the enhanced RAG system"""
    try:
        with st.spinner("🔄 Initializing Enhanced AI Analytics... This may take a moment..."):
            return EnhancedDonorRAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize enhanced RAG system: {e}")
        return None


def display_enhanced_header():
    """Display the enhanced header"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<h1 class="main-header">📊 Enhanced Donor Analytics RAG System</h1>', unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="enhanced-badge">INTERMEDIATE ANALYTICS</span>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced AI-powered donor analytics with trend analysis, segmentation, and predictive insights
        </p>
        <p style="color: #888;">
            🚀 Enhanced with time-based analysis, retention tracking, and donor segmentation
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_enhanced_metrics(rag_system):
    """Display enhanced metrics with analytics"""
    st.sidebar.markdown("## 📊 Enhanced Analytics Dashboard")

    try:
        # Basic metrics
        total_donors = rag_system.execute_sql_query("SELECT COUNT(*) as total FROM donors;")
        total_amount = rag_system.execute_sql_query("SELECT SUM(amount) as total FROM donations;")

        if not total_donors.empty and not total_amount.empty:
            st.sidebar.metric("👥 Total Donors", f"{total_donors.iloc[0]['total']:,}")
            st.sidebar.metric("💵 Total Raised", f"${total_amount.iloc[0]['total']:,.2f}")

        # Enhanced analytics
        st.sidebar.markdown("### 🔍 Quick Analytics")

        # Current year performance
        current_year_data = rag_system.execute_sql_query("""
            SELECT 
                COUNT(*) as donations,
                SUM(amount) as amount,
                COUNT(DISTINCT donor_id) as unique_donors
            FROM donations 
            WHERE EXTRACT(YEAR FROM donation_date) = EXTRACT(YEAR FROM CURRENT_DATE);
        """)

        if not current_year_data.empty:
            row = current_year_data.iloc[0]
            st.sidebar.metric("🗓️ This Year Donations", f"{row['donations']:,}")
            st.sidebar.metric("💰 This Year Amount", f"${row['amount']:,.0f}")
            st.sidebar.metric("👤 Active Donors", f"{row['unique_donors']:,}")

        # Retention rate preview
        if st.sidebar.button("🔄 Calculate Retention Rate", use_container_width=True):
            with st.spinner("Calculating retention..."):
                retention_data = rag_system.analytics.calculate_retention_rate()
                if not retention_data.empty:
                    latest_retention = retention_data.iloc[-1]['retention_rate_percent']
                    st.sidebar.success(f"Latest Retention Rate: {latest_retention}%")

        # Quick segmentation
        if st.sidebar.button("📈 Donor Segments Preview", use_container_width=True):
            with st.spinner("Analyzing segments..."):
                segments = rag_system.execute_sql_query("""
                    SELECT 
                        CASE 
                            WHEN lifetime_value >= 10000 THEN 'Major Donors'
                            WHEN total_donations >= 10 THEN 'Frequent Donors'
                            WHEN EXTRACT(DAYS FROM (CURRENT_DATE - last_donation_date)) > 365 THEN 'Lapsed'
                            ELSE 'Regular'
                        END as segment,
                        COUNT(*) as count
                    FROM donors
                    GROUP BY segment;
                """)
                if not segments.empty:
                    for _, row in segments.iterrows():
                        st.sidebar.write(f"**{row['segment']}:** {row['count']}")

    except Exception as e:
        st.sidebar.error(f"Error loading enhanced metrics: {e}")


def display_enhanced_sample_questions():
    """Display categorized sample questions"""
    st.sidebar.markdown("### 💡 Enhanced Sample Questions")

    # Categorize questions by complexity
    question_categories = {
        "📊 Basic Analytics": [
            "Show me total donations for 2024",
            "Who are our top 10 donors by lifetime value?",
            "What's our average donation amount this year?",
            "How many donors gave more than $1000?"
        ],
        "📈 Trend Analysis": [
            "Compare donation trends between Q1 and Q2",
            "Show me monthly donation patterns this year",
            "Analyze year-over-year growth",
            "What's our donation trend over the past 3 years?"
        ],
        "🎯 Donor Analytics": [
            "Analyze donor retention rates over the past 3 years",
            "Segment our donors by giving behavior",
            "Show me donation patterns by age group",
            "What's our new vs returning donor ratio?"
        ],
        "💰 Channel Analysis": [
            "Which fundraising channels are most effective?",
            "Compare online vs offline donation performance",
            "Show me donation amounts by payment method",
            "Analyze channel performance over time"
        ]
    }

    for category, questions in question_categories.items():
        with st.sidebar.expander(category, expanded=False):
            for question in questions:
                if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                    st.session_state.sample_question = question


def create_enhanced_visualization(df, query_type, question=""):
    """Create enhanced visualizations based on query type and data"""
    if df.empty:
        return None

    try:
        # Trend analysis visualizations
        if query_type == 'trend_analysis' or 'trend' in question.lower():
            if 'quarter' in df.columns and 'total_amount' in df.columns:
                fig = px.line(df, x='quarter', y='total_amount',
                              title='Quarterly Donation Trends',
                              labels={'total_amount': 'Total Amount ($)', 'quarter': 'Quarter'})
                fig.update_traces(mode='lines+markers')
                return fig
            elif 'month' in df.columns and 'total_amount' in df.columns:
                fig = px.bar(df, x='month', y='total_amount',
                             title='Monthly Donation Trends',
                             labels={'total_amount': 'Total Amount ($)', 'month': 'Month'})
                return fig

        # Retention analysis visualization
        elif query_type == 'retention_analysis':
            if 'year' in df.columns and 'retention_rate_percent' in df.columns:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add retention rate line
                fig.add_trace(
                    go.Scatter(x=df['year'], y=df['retention_rate_percent'],
                               mode='lines+markers', name='Retention Rate (%)',
                               line=dict(color='red', width=3)),
                    secondary_y=False,
                )

                # Add donor count bars
                fig.add_trace(
                    go.Bar(x=df['year'], y=df['total_donors'],
                           name='Total Donors', opacity=0.7),
                    secondary_y=True,
                )

                fig.update_xaxes(title_text="Year")
                fig.update_yaxes(title_text="Retention Rate (%)", secondary_y=False)
                fig.update_yaxes(title_text="Number of Donors", secondary_y=True)
                fig.update_layout(title_text="Donor Retention Analysis")

                return fig

        # Segmentation visualization
        elif query_type == 'segmentation':
            if 'donor_segment' in df.columns:
                segment_counts = df['donor_segment'].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                             title='Donor Segmentation Distribution')
                return fig

        # Acquisition analysis
        elif query_type == 'acquisition_analysis':
            if 'new_donors' in df.columns and 'returning_donors' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df['month'], y=df['new_donors'],
                                     name='New Donors', marker_color='lightblue'))
                fig.add_trace(go.Bar(x=df['month'], y=df['returning_donors'],
                                     name='Returning Donors', marker_color='darkblue'))
                fig.update_layout(title='New vs Returning Donors by Month',
                                  xaxis_title='Month', yaxis_title='Number of Donors',
                                  barmode='stack')
                return fig

        # Standard visualizations for other cases
        elif 'amount' in df.columns or 'total_amount' in df.columns:
            amount_col = 'total_amount' if 'total_amount' in df.columns else 'amount'

            if 'donor_name' in df.columns:
                fig = px.bar(df.head(15), x='donor_name', y=amount_col,
                             title='Top Donors by Amount')
                fig.update_xaxes(tickangle=45)
                return fig
            elif 'channel' in df.columns:
                fig = px.bar(df, x='channel', y=amount_col,
                             title='Donations by Channel')
                return fig
            elif 'age_range' in df.columns:
                fig = px.bar(df, x='age_range', y=amount_col,
                             title='Donations by Age Range')
                return fig

        elif 'lifetime_value' in df.columns:
            fig = px.bar(df.head(15), x='donor_name', y='lifetime_value',
                         title='Top Donors by Lifetime Value')
            fig.update_xaxes(tickangle=45)
            return fig

        return None

    except Exception as e:
        st.error(f"Error creating enhanced visualization: {e}")
        return None


def display_query_insights(result):
    """Display analytical insights for the query"""
    if 'insights' in result and result['insights']:
        st.markdown(f'<div class="insight-box"><strong>📊 Analytical Insights:</strong><br>{result["insights"]}</div>',
                    unsafe_allow_html=True)

    if 'query_type' in result:
        query_type = result['query_type'].replace('_', ' ').title()
        st.markdown(f'<div class="query-type-badge">Analysis Type: {query_type}</div>',
                    unsafe_allow_html=True)


def main():
    """Enhanced main application function"""

    # Display enhanced header
    display_enhanced_header()

    # Initialize enhanced RAG system
    rag_system = initialize_enhanced_rag_system()

    if not rag_system:
        st.error("❌ Failed to initialize the enhanced system.")
        st.info("Make sure PostgreSQL is running and Ollama is available with the mistral:7b model.")
        return

    # Display enhanced metrics
    display_enhanced_metrics(rag_system)

    # Display enhanced sample questions
    display_enhanced_sample_questions()

    # Main query interface
    st.markdown("## 🔍 Advanced Donor Analytics Query")

    # Handle sample question selection
    default_question = ""
    if 'sample_question' in st.session_state:
        default_question = st.session_state.sample_question
        del st.session_state.sample_question

    # Enhanced query input with examples
    st.markdown("### Ask sophisticated questions about your donor data:")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Enter your analytical question:",
            value=default_question,
            placeholder="e.g., Compare donation trends between Q1 and Q2 with growth analysis",
            help="Try complex questions like trend analysis, donor segmentation, or retention rates!"
        )

    with col2:
        analysis_type = st.selectbox(
            "Analysis Focus:",
            ["Auto-detect", "Trend Analysis", "Donor Segmentation", "Retention Analysis", "Channel Performance"],
            help="Hint for the AI about what type of analysis you want"
        )

    # Enhanced query execution
    if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True) and query:

        try:
            with st.spinner("🧠 Running advanced analytics... This may take a moment..."):
                result = rag_system.ask_question(query)

            if isinstance(result, dict):
                # Display query type and insights
                display_query_insights(result)

                # Display enhanced answer
                st.markdown("## 💬 Analytics Results")
                st.markdown(f'<div class="analytics-container">{result["answer"]}</div>',
                            unsafe_allow_html=True)

                # Enhanced results tabs
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📊 Data Results", "📈 Advanced Visualization", "🔧 SQL Query", "🎯 Recommendations"])

                with tab1:
                    st.markdown("### Raw Analytics Data")
                    if not result['raw_results'].empty:
                        st.dataframe(result['raw_results'], use_container_width=True)

                        # Enhanced summary statistics
                        numeric_cols = result['raw_results'].select_dtypes(include=['float64', 'int64']).columns
                        if len(numeric_cols) > 0:
                            st.markdown("#### Statistical Summary")
                            st.dataframe(result['raw_results'][numeric_cols].describe())

                        # Data export option
                        csv = result['raw_results'].to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"donor_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No data returned for this query.")

                with tab2:
                    st.markdown("### Advanced Data Visualization")
                    if not result['raw_results'].empty:
                        query_type = result.get('query_type', 'standard')
                        fig = create_enhanced_visualization(result['raw_results'], query_type, query)

                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                            # Additional chart options
                            st.markdown("#### Alternative Views")
                            chart_type = st.selectbox(
                                "Chart Type:",
                                ["Auto", "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"],
                                key="chart_selector"
                            )

                            if chart_type != "Auto":
                                # Create custom chart based on selection
                                # Implementation would depend on data structure
                                st.info(f"Custom {chart_type} view - Feature coming soon!")
                        else:
                            st.info("No suitable visualization available for this data type.")

                            # Show data preview instead
                            if len(result['raw_results']) > 0:
                                st.write("**Data Preview:**")
                                st.write(f"• {len(result['raw_results'])} records returned")
                                st.write(f"• Columns: {', '.join(result['raw_results'].columns)}")
                    else:
                        st.info("No data to visualize.")

                with tab3:
                    st.markdown("### Generated SQL Query")
                    st.code(result["sql_query"], language="sql")

                    # Query performance info
                    if not result['raw_results'].empty:
                        st.success(f"✅ Query executed successfully - {len(result['raw_results'])} rows returned")
                    else:
                        st.warning("⚠️ Query executed but returned no results")

                    # SQL optimization suggestions
                    if len(result['raw_results']) > 1000:
                        st.info(
                            "💡 Large result set detected. Consider adding date filters or LIMIT clauses for better performance.")

                with tab4:
                    st.markdown("### Strategic Recommendations")

                    # Generate recommendations based on query type and results
                    if result.get('query_type') == 'retention_analysis' and not result['raw_results'].empty:
                        st.markdown("""
                        **📈 Retention Strategy Recommendations:**
                        - Focus on donors with declining retention trends
                        - Implement re-engagement campaigns for lapsed donors
                        - Analyze successful retention factors from high-performing periods
                        """)
                    elif result.get('query_type') == 'segmentation' and not result['raw_results'].empty:
                        st.markdown("""
                        **🎯 Segmentation Action Items:**
                        - Develop targeted campaigns for each donor segment
                        - Create personalized communication strategies
                        - Set segment-specific fundraising goals and metrics
                        """)
                    elif 'trend' in query.lower() and not result['raw_results'].empty:
                        st.markdown("""
                        **📊 Trend Analysis Insights:**
                        - Identify seasonal patterns for campaign timing
                        - Focus resources on high-performing periods
                        - Address declining trends with targeted interventions
                        """)
                    else:
                        st.markdown("""
                        **💡 General Recommendations:**
                        - Use these insights to inform your fundraising strategy
                        - Consider running follow-up analysis on interesting patterns
                        - Share findings with your development team for action planning
                        """)

                    # Quick follow-up questions
                    st.markdown("#### 🔄 Suggested Follow-up Questions:")
                    follow_up_questions = [
                        "Analyze this data by different time periods",
                        "Compare with previous year's performance",
                        "Show demographic breakdown of these results",
                        "Identify top performers in this analysis"
                    ]

                    for fq in follow_up_questions:
                        if st.button(fq, key=f"followup_{hash(fq)}", use_container_width=True):
                            st.session_state.sample_question = fq

            else:
                st.error(f"❌ {result}")

        except Exception as e:
            st.error(f"❌ Error processing advanced query: {str(e)}")
            st.error("Please check the console for detailed error information.")

    # Enhanced info section
    with st.expander("ℹ️ About Enhanced Analytics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 Enhanced Capabilities:**
            - **Time-based Analysis**: "Show me Q1 vs Q2 trends"
            - **Donor Segmentation**: RFM analysis and behavior patterns
            - **Retention Tracking**: Multi-year retention rate analysis
            - **Trend Analysis**: Growth rates and comparative insights
            - **Channel Performance**: Multi-channel effectiveness analysis
            """)

        with col2:
            st.markdown("""
            **🔧 Technical Enhancements:**
            - **Advanced SQL Generation**: Complex time-based queries
            - **Statistical Analysis**: Growth rates, percentages, trends
            - **Smart Visualizations**: Context-aware chart selection
            - **Analytical Insights**: Automated pattern detection
            - **Strategic Recommendations**: Actionable insights
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🚀 Enhanced with Advanced Analytics | Powered by Mistral 7B + PostgreSQL | 🔒 100% Local & Private</p>
        <p>📊 Intermediate Analytics: Trend Analysis • Donor Segmentation • Retention Tracking</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    from datetime import datetime

    main()