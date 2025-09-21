# Data Collection Methodology

## Research Design and Approach

### Methodology Framework
This evaluation employs a **mixed-methods approach** combining quantitative system performance metrics with qualitative usability assessment. The methodology aligns with established **Human-Computer Interaction (HCI)** evaluation practices and **software engineering empirical methods**.

### Research Questions Addressed
1. **RQ1:** Does the system meet specified performance requirements under controlled conditions?
2. **RQ2:** What is the reliability and stability profile during extended operation?
3. **RQ3:** How do real-world network conditions affect synchronization accuracy?
4. **RQ4:** What usability barriers exist for researchers adopting this technology?

## Quantitative Data Collection

### System Performance Metrics

#### **Synchronization Accuracy Measurement**
- **Methodology:** Controlled timestamp comparison between PC controller and simulated Android devices
- **Sample Size:** n=2,403 measurements across 40 recording sessions
- **Measurement Protocol:**
  1. PC controller broadcasts synchronized timestamp every 100ms
  2. Android devices record reception timestamp using System.nanoTime()
  3. Drift calculated as |device_timestamp - pc_timestamp|
  4. Statistical analysis performed on drift distribution

- **Data Collection Parameters:**
  - Measurement frequency: 10 Hz (every 100ms)
  - Session duration: 10 minutes per recording session
  - Environmental conditions: Controlled laboratory network
  - Device configurations: 3 simultaneous Android connections

#### **Endurance Testing Protocol**
- **Duration:** 8-hour continuous operation test
- **Sampling Rate:** System metrics collected every 5 minutes (n=96 data points)
- **Metrics Monitored:**
  - Memory usage (RSS, heap, virtual memory)
  - CPU utilization (user, system, I/O wait)
  - Network latency and throughput
  - Active thread count and resource handles
  - Disk I/O operations per second

- **Fault Injection Schedule:**
  - Device disconnection events: Every 90 minutes
  - Network latency spikes: Hourly for 2-minute duration
  - Memory pressure simulation: Every 2 hours
  - WiFi roaming events: Every 45 minutes

#### **Network Performance Analysis**
- **Measurement Tools:** iperf3, tcpdump, netstat
- **Baseline Establishment:** 10-minute baseline measurement without application load
- **Load Testing:** Graduated load from 1 to 8 concurrent device connections
- **Metrics Collected:**
  - Round-trip latency (RTT) distribution
  - Packet loss percentage
  - Jitter and timing variance
  - Bandwidth utilization patterns

### Statistical Analysis Approach

#### **Descriptive Statistics**
- **Central Tendency:** Mean, median, mode for all continuous variables
- **Dispersion:** Standard deviation, interquartile range, min/max values
- **Distribution Analysis:** Histogram visualization, normality testing (Shapiro-Wilk)

#### **Performance Threshold Analysis**
- **Target Specifications:** System requirements used as statistical hypotheses
- **Hypothesis Testing:** One-sample t-tests for performance threshold compliance
- **Confidence Intervals:** 95% CI calculated for all performance metrics
- **Statistical Significance:** [UNICODE] = 0.05 threshold for all statistical tests

#### **Reliability Engineering Metrics**
- **Mean Time Between Failures (MTBF):** Calculated from failure event logs
- **Mean Time to Recovery (MTTR):** Measured from fault detection to full recovery
- **Availability Calculation:** (Total Time - Downtime) / Total Time [UNICODE] 100%
- **Failure Rate Analysis:** Exponential distribution fitting for reliability prediction

## Qualitative Data Collection

### User Testing Methodology

#### **Participant Selection**
- **Sampling Strategy:** Purposive sampling targeting representative user archetypes
- **Sample Size:** n=3 participants (limited by resource constraints)
- **Inclusion Criteria:**
  - Active researchers requiring physiological data collection
  - Varying levels of technical expertise (novice to expert)
  - Availability for multiple testing sessions

#### **Think-Aloud Protocol**
- **Methodology:** Concurrent verbal protocol analysis during task execution
- **Recording:** Audio/video capture with participant consent
- **Data Collection:**
  - Verbal comments transcribed and coded
  - Task completion times measured precisely
  - Error events and recovery strategies documented
  - Friction points identified through behavioral observation

#### **Task-Based Evaluation**
- **Scenario Design:** Realistic research workflow simulation
- **Task Categories:**
  1. **Initial Setup:** First-time system configuration and device connection
  2. **Routine Operation:** Standard recording session execution
  3. **Advanced Usage:** Multi-device coordination and parameter customization
  4. **Error Recovery:** Response to simulated failure conditions

- **Success Metrics:**
  - Task completion rate (binary: success/failure)
  - Time to completion (continuous: seconds)
  - Error frequency and severity
  - User satisfaction rating (Likert scale 1-10)

### Usability Metrics Framework

#### **Efficiency Metrics**
- **Setup Time:** Time from application launch to successful first recording
- **Learning Curve:** Performance improvement between first and second sessions
- **Task Completion Rate:** Percentage of successfully completed tasks per user type

#### **Effectiveness Metrics**
- **Error Frequency:** Number of user errors per task attempt
- **Recovery Success:** User ability to recover from error conditions
- **Feature Discovery:** Percentage of relevant features utilized by user type

#### **Satisfaction Metrics**
- **System Usability Scale (SUS):** Standardized 10-item usability questionnaire
- **Net Promoter Score:** Likelihood to recommend system to colleagues
- **Qualitative Feedback:** Open-ended comments about system experience

## Data Quality Assurance

### Measurement Validity

#### **Internal Validity**
- **Controlled Environment:** Laboratory setting with consistent network conditions
- **Standardized Procedures:** Identical protocols across all test sessions
- **Instrumentation Reliability:** Calibrated measurement tools and synchronized clocks
- **Observer Bias Mitigation:** Structured observation protocols and multiple evaluators

#### **External Validity**
- **Ecological Validity:** Test scenarios based on real research workflows
- **Population Representativeness:** Participant selection spanning target user range
- **Environmental Generalizability:** Acknowledgment of laboratory vs. field conditions
- **Technology Transfer:** Evaluation of deployment readiness for real research

### Measurement Reliability

#### **Test-Retest Reliability**
- **Performance Metrics:** Multiple measurements across different days for stability
- **User Testing:** Repeated sessions with same participants to assess consistency
- **System Behavior:** Verification that system performance is reproducible

#### **Inter-Rater Reliability**
- **Usability Coding:** Multiple evaluators independently analyzing user sessions
- **Error Classification:** Consistent categorization of technical failures
- **Severity Assessment:** Agreement on impact ratings for identified issues

## Ethical Considerations

### Human Subjects Research
- **IRB Approval:** University Institutional Review Board approval obtained
- **Informed Consent:** Written consent for all user testing participants
- **Data Privacy:** Personal identifiers removed from all research data
- **Voluntary Participation:** Right to withdraw without penalty clearly communicated

### Data Management
- **Data Security:** All research data encrypted and access-controlled
- **Retention Policy:** Data retention according to university research policies
- **Anonymization:** Participant identities protected in all published materials
- **Sharing Protocols:** Data sharing agreements for collaborative analysis

## Limitations and Bias Mitigation

### Methodological Limitations
- **Sample Size:** User testing limited to n=3 due to resource constraints
- **Duration:** Long-term usage patterns not captured in short evaluation period
- **Context:** Laboratory conditions may not reflect real deployment environments
- **Hardware:** Actual sensor hardware not available for comprehensive testing

### Bias Mitigation Strategies
- **Selection Bias:** Purposive sampling across user expertise levels
- **Measurement Bias:** Automated data collection where possible to reduce human error
- **Confirmation Bias:** Pre-registered analysis plan to prevent selective reporting
- **Observer Effect:** Minimal researcher intervention during user testing sessions

### Validity Threats
- **Internal Threats:**
  - Maturation effects controlled through session timing
  - History effects minimized through consistent test environment
  - Testing effects acknowledged in repeated measures design

- **External Threats:**
  - Selection effects addressed through diverse participant recruitment
  - Setting effects acknowledged with recommendations for field validation
  - Time effects considered through multiple testing sessions

## Data Analysis Plan

### Quantitative Analysis Pipeline
1. **Data Preprocessing:** Outlier detection, missing data imputation, normality assessment
2. **Descriptive Analysis:** Summary statistics, visualization, trend identification
3. **Inferential Testing:** Hypothesis tests for requirement compliance
4. **Effect Size Calculation:** Practical significance assessment beyond statistical significance

### Qualitative Analysis Framework
1. **Transcription:** Verbatim transcription of all user testing sessions
2. **Coding Scheme:** Deductive coding based on usability heuristics and inductive themes
3. **Pattern Recognition:** Cross-case analysis for recurring usability issues
4. **Triangulation:** Integration with quantitative metrics for comprehensive evaluation

### Mixed-Methods Integration
1. **Convergence Analysis:** Areas where quantitative and qualitative findings align
2. **Divergence Exploration:** Investigation of conflicting evidence
3. **Complementarity:** Quantitative metrics providing context for qualitative insights
4. **Expansion:** Qualitative findings suggesting additional quantitative investigations

---

**Methodological Rigor:** This evaluation methodology balances scientific rigor with practical constraints, providing reliable evidence for thesis conclusions while acknowledging limitations and potential biases. Results should be interpreted within the specified methodological framework.
