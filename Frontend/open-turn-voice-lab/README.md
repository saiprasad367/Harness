# Open-Turn Voice Lab

### Adaptive, Accessibility-Aware Agent Harness for Full-Duplex Voice Models

Build a complete, premium, highly interactive frontend from scratch for my research project:

**“An Adaptive, Accessibility-Aware Agent Harness for Full-Duplex Voice Models.”**

The frontend should represent a sophisticated AI research platform that combines voice interaction, adaptive turn-taking, speech analysis, accessibility settings, dataset exploration, human annotation, simulation, and evaluation.

## 1. Technology and Project Requirements

- React + TypeScript + Vite
- Tailwind CSS
- Modern reusable UI components
- Lucide icons
- Recharts for charts
- Responsive desktop, tablet, and mobile layouts
- Frontend only; backend integration will be implemented separately.
- Use realistic mock data for UI development, clearly distinguishable from actual model results.
- Create a well-organized, scalable frontend architecture.
- Do not build a simple landing page. Build a complete multi-page application.

## 2. Overall Design and Theme

Use my attached UI reference image as the primary visual inspiration.

Create a premium AI research dashboard with:

- Light theme
- White and very light lavender backgrounds
- Primary purple: `#7C3AED`
- Background: `#F8F9FC`
- Cards: `#FFFFFF`
- Text: `#111827`
- Secondary text: `#6B7280`
- Subtle borders, soft shadows, rounded corners
- Professional typography
- Clean spacing and elegant layouts
- Smooth transitions and subtle animations
- High-quality charts, audio visualizations, and interactive elements

Avoid excessive gradients, clutter, oversized headings, and unnecessary decorative elements.

## 3. Application Layout

Create a professional application shell.

### Sidebar Navigation

Include:

- Project logo and name
- Overview
- Live Voice Studio
- Simulation Studio
- Dataset Explorer
- Ground-Truth Annotation
- Evaluation & Benchmarks
- Research Reports
- Settings

The sidebar should support collapsing and expanding, active-page highlighting, tooltips, and responsive mobile navigation.

### Top Navigation Bar

Include:

- Current page title and breadcrumb
- Global search
- Notification icon
- Help/documentation icon
- Theme control
- User profile/avatar
- System connection status
- Current session indicator

Use consistent navigation throughout all pages.

---

# 4. Overview Dashboard

Create a comprehensive research dashboard that provides a quick overview of the entire system.

### Header

- Welcome/title section
- Project description
- Current system status
- Last updated timestamp
- Quick actions: Start Voice Session, Run Simulation, Annotate Audio

### Statistics Cards

Display:

- Total audio recordings
- Total speakers
- Total candidate pauses
- Annotated pauses
- Annotation completion percentage
- Current model status
- Current active voice sessions
- Evaluation status

Include small trend indicators and progress bars where appropriate.

### System Architecture Visualization

Create a clean interactive pipeline diagram:

Microphone Input → Audio Preprocessing → Silero VAD → Pause Detection → Feature Extraction → Turn Classification → Adaptive Turn Controller → Moshi Voice Model → Agent Audio Output.

Show each component as a connected node with:

- Component name
- Status indicator
- Processing stage
- Input/output information
- Latency display
- Error or warning states

Use different colors for active, inactive, processing, and disconnected components.

### Live Voice Interaction Panel

Create a prominent live interaction card featuring:

- Start/stop microphone
- Start/stop session
- Microphone permission status
- Input audio level meter
- Live waveform
- Speech detection status
- Pause duration
- VAD confidence
- Turn intention: END / CONTINUE / UNCERTAIN
- Adaptive controller decision
- Agent response status

### Research Activity

Include:

- Recent sessions
- Recent annotations
- Latest evaluation runs
- Recent simulation activity
- System event timeline

### Dashboard Charts

Display:

- Dataset distribution
- Speaker/cohort distribution
- Annotation progress
- Turn decision distribution
- Recent evaluation performance

Use interactive charts with tooltips, filters, and legends.

---

# 5. Live Voice Studio

This is the primary page for real-time voice interaction.

Create a polished, immersive voice workspace with a professional research interface.

### Session Configuration

Include:

- Microphone device selector
- Audio input configuration
- Sample rate display
- Audio channel information
- Voice model selector
- Moshi connection status
- Mock Mode / Live Mode selector
- Accessibility mode selector:
  - Patient
  - Normal
  - Quick
- Session configuration panel

### Voice Session Controls

Include:

- Connect to model
- Disconnect
- Start session
- Pause session
- Resume session
- End session
- Reset session
- Mute/unmute microphone
- Interrupt agent
- Clear conversation

### Live Audio Visualization

Create a professional audio visualization area containing:

- User microphone waveform
- Agent output waveform
- Input audio level meter
- Agent audio level meter
- Spectrogram visualization
- Speech activity regions
- Detected pause regions
- Current audio timestamp
- Streaming status

### Adaptive Turn-Taking Panel

Display:

- Current VAD state
- Speech probability
- Pause duration
- Extracted pause features
- Classifier output
- Predicted turn intention
- Controller decision
- Active accessibility mode
- Current threshold or sensitivity
- Decision explanation

Show the distinction between:

- END — User has likely completed their turn
- CONTINUE — User may continue speaking
- UNCERTAIN — More information or waiting may be necessary

### Conversation Timeline

Create a real-time conversation timeline with:

- User speech events
- Pause events
- END/CONTINUE/UNCERTAIN decisions
- Agent responses
- Interruptions
- Session events
- Timestamps
- Event details

Allow filtering by event type.

### Full-Duplex Interaction

Include UI indicators for:

- Simultaneous user and agent audio
- Agent speaking state
- User speaking state
- Barge-in/interruption state
- Turn ownership
- Audio streaming state

Clearly distinguish mock behavior from actual live model connectivity.

---

# 6. Simulation Studio

Create a dedicated environment for testing adaptive turn-taking behavior without requiring live microphone interaction.

### Simulation Configuration

Include:

- Audio recording selector
- Upload audio interface
- Dataset selection
- Speaker selection
- Accessibility mode selector
- Adaptive controller toggle
- Fixed-threshold baseline toggle
- Simulation speed control
- Playback speed
- Start, pause, stop, reset, and replay controls

### Simulation Visualization

Display:

- Audio waveform
- Playback progress
- Speech segments
- VAD regions
- Pause boundaries
- Pause duration labels
- Predicted turn decisions
- Adaptive threshold visualization
- Fixed threshold comparison

### Simulation Timeline

Display a chronological event timeline showing:

- Speech start
- Speech end
- Detected pauses
- Classifier decisions
- Controller actions
- Simulated agent responses

Allow clicking an event to inspect its details.

### Comparison Panel

Compare:

- Adaptive turn controller
- Fixed-threshold baseline

Display:

- Number of END decisions
- Number of CONTINUE decisions
- Number of UNCERTAIN decisions
- False interruptions
- Delayed responses
- Decision timing
- Average response delay

Only show metrics when they are available from actual data or clearly marked simulation data.

---

# 7. Dataset Explorer

Create a complete research dataset management and exploration interface.

### Dataset Overview

Display:

- Dataset name
- Dataset description
- Total recordings
- Number of speakers
- Cohort distribution
- Dataset split distribution
- Audio duration statistics
- Sampling rate information

### Recording Table

Include:

- Recording ID
- Speaker ID
- Cohort
- Dataset split
- Recording duration
- Audio format
- Annotation status
- Number of detected pauses
- Actions

Add:

- Search
- Sorting
- Pagination
- Filters
- Multi-select
- Export controls

### Filters

Support filtering by:

- Speaker
- Cohort
- Train/validation/test split
- Recording duration
- Annotation status
- Pause count

### Recording Details

When a recording is selected, open a detailed view containing:

- Recording metadata
- Audio player
- Waveform visualization
- Spectrogram
- Playback speed controls
- VAD visualization
- Pause boundaries
- Pause duration markers
- Speaker metadata
- Annotation history

Allow zooming and selecting specific time ranges.

### Audio Upload

Provide a polished drag-and-drop upload interface.

Include:

- File selection
- Supported format information
- Upload progress
- Validation status
- Processing status
- Error handling

The interface should be ready for future backend integration.

---

# 8. Ground-Truth Annotation Studio

Create a specialized interface for human annotation of pause and turn-taking behavior.

This page should feel like a professional audio annotation tool.

### Annotation Overview

Display:

- Total candidate pauses
- Annotated pauses
- Remaining pauses
- Annotation progress
- Annotator activity
- Annotation completion rate

### Candidate Pause Queue

Include:

- Candidate ID
- Recording ID
- Speaker ID
- Cohort
- Pause duration
- Annotation status
- Confidence
- Priority

Support searching, filtering, sorting, and selecting candidates.

### Annotation Workspace

Create a large central audio workspace containing:

- Audio playback
- Waveform
- Spectrogram
- Speech segments
- Pause boundary markers
- Pause start and end timestamps
- Playback speed
- Zoom controls
- Timeline navigation

Highlight the selected pause region.

### Annotation Controls

Provide three prominent annotation options:

- END
- CONTINUE
- UNCERTAIN

Include:

- Keyboard shortcuts
- Annotator confidence selector
- Notes field
- Optional explanation
- Save annotation
- Save and next
- Skip candidate
- Previous/next candidate
- Undo annotation

### Annotation Details

Display:

- Recording information
- Pause duration
- Model prediction
- Classifier confidence
- Annotator decision
- Previous annotation history

Clearly distinguish model predictions from human ground-truth labels.

### Annotation Progress

Include:

- Progress bar
- Completed/remaining count
- Annotation history
- Annotator agreement visualization
- Disagreement review interface

Do not generate fake human annotations or claim that mock labels are ground truth.

---

# 9. Evaluation & Benchmarks

Create a comprehensive evaluation dashboard for research results.

### Evaluation Overview

Display:

- Evaluation run ID
- Model name
- Dataset split
- Evaluation timestamp
- Evaluation status
- Number of evaluated samples

### Performance Metrics

Include cards for:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- False Interruption Rate (FIR)
- False Wait Rate (FWR)
- Average decision latency

Each metric should have:

- Value
- Definition tooltip
- Comparison with baseline
- Trend indicator when historical data exists

### Interactive Charts

Include:

- Confusion matrix
- ROC curve
- Precision-recall curve
- Model performance comparison
- FIR/FWR comparison
- Latency distribution
- Decision distribution
- Speaker-level performance
- Cohort-level performance

### Evaluation Filters

Allow filtering by:

- Dataset
- Cohort
- Speaker
- Model
- Accessibility mode
- Evaluation run

### Baseline Comparison

Compare:

- Adaptive model
- Fixed-threshold baseline
- Other available model versions

Include a detailed comparison table.

### Evaluation History

Display previous evaluation runs with:

- Run ID
- Model version
- Dataset
- Metrics
- Date
- Status
- View details action

Include export options for evaluation results.

---

# 10. Research Reports

Create a dedicated research reporting page.

### Report Categories

- Dataset reports
- Annotation reports
- Model evaluation reports
- Simulation reports
- Accessibility analysis reports
- System performance reports

### Report Interface

Include:

- Report title
- Report type
- Creation date
- Report status
- Summary
- Key findings
- Associated dataset
- Associated evaluation run

### Report Details

Provide a detailed report viewer with:

- Executive summary
- Dataset information
- Methodology
- Annotation statistics
- Evaluation results
- Charts and visualizations
- Limitations
- Conclusions

### Export Options

Include:

- PDF export
- CSV export
- JSON export
- Download report
- Print report

Create polished report layouts suitable for academic research.

---

# 11. Settings and Accessibility

Create a comprehensive settings page.

### Audio Settings

Include:

- Microphone selection
- Input sample rate
- Audio channel selection
- Input volume
- Output volume
- Audio visualization settings
- Noise filtering preferences

### Voice Model Settings

Include:

- Model selection
- Connection settings UI
- Session timeout
- Streaming preferences
- Mock/Live mode
- Connection status

Do not expose or hardcode API keys.

### Accessibility Settings

Include:

- Patient mode
- Normal mode
- Quick mode
- Pause sensitivity
- Waiting duration
- Response delay preferences
- Visual indicators
- Reduced motion
- Keyboard navigation
- High contrast option
- Audio visualization preferences

Explain each accessibility mode with a short description.

### Interface Settings

Include:

- Theme selection
- Sidebar preferences
- Notification settings
- Chart display settings
- Default dataset
- Default evaluation configuration

---

# 12. Additional Features

Include these supporting features throughout the application:

- Global search
- Notifications
- Recent activity
- Session history
- Event logs
- Empty states
- Loading skeletons
- Error states
- Confirmation dialogs
- Tooltips
- Toast notifications
- Data export interfaces
- Breadcrumb navigation
- Responsive layouts
- Keyboard shortcuts
- Accessible controls

Every button, dropdown, filter, slider, tab, navigation item, and form should work meaningfully in the frontend.

Use realistic frontend mock interactions where backend functionality is not yet available.

---

# 13. Component and Data Architecture

Build reusable components for:

- Sidebar
- Navbar
- Metric cards
- Audio player
- Waveform visualization
- Spectrogram
- Timeline
- Event log
- Data tables
- Charts
- Annotation controls
- Session controls
- Status indicators
- Dialogs
- Filters
- Settings panels

Create typed data structures and clean page/component separation.

Prepare frontend service interfaces and API integration points for the backend to be connected later.

Keep mock data isolated from real backend data.

---

# 14. Final Deliverable

Build the complete frontend with all pages, navigation, visualizations, controls, and interactive functionality.

The final application should look and feel like a premium AI research platform used by researchers working on speech processing, accessibility, and full-duplex voice systems.

Prioritize:
1. Complete multi-page application
2. Excellent UI design
3. Accurate representation of research functionality
4. Interactive audio visualization
5. Clear turn-taking and accessibility concepts
6. Responsive layout
7. Backend-ready architecture

Use my attached reference image for visual inspiration, but expand it into a complete, professionally designed research platform.

Do not stop after creating the dashboard. Implement every page and make the application cohesive, polished, and fully navigable.

## Architecture & Research Deployment

The Open-Turn Voice Lab frontend connects to:
- **FastAPI / Starlette Backend** (`api_server.py`) for Silero VAD, adaptive Logistic Regression turn classifier, and TORGO corpus data.
- **Kyutai Moshi 7B Full-Duplex Model** running on remote Kaggle Dual T4 GPU via WebSockets.
- Fully compatible with Vercel deployment for frontend and Render for backend.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
