# 📱 MediCheck

**MediCheck** is an iOS app that empowers users to manage their medications effectively by tracking usage, detecting potential drug interactions, and scheduling timely dosage reminders. By combining cutting-edge OCR, machine learning, and reliable FDA data, MediCheck delivers real-time, personalized insights for safer medication management.

---

## Key Features

- **User Enrollment:**  
  - **Registration:** Users quickly sign up by providing essential details such as Date of Birth, gender, and pregnancy status.  
  - **Medication History:** Capture current medications to create a personalized health profile.  
  - **Data Security:** All user information is securely stored and managed with MongoDB, ensuring privacy and compliance.

- **Medication Recognition:**  
  - **Image Capture:** Users take a photo of the medication packaging or the pill itself using the device’s camera.  
  - **OCR Processing:** Apple VisionKit extracts text from the image for further processing.  
  - **NLP Integration:** Python’s Sentence Transformer converts the extracted text into semantic embeddings.  
  - **Machine Learning Matching:** A combination of semantic and lexical similarity algorithms ensures accurate medication identification using learned weights.

- **FDA API Integration:**  
  - **Comprehensive Data Retrieval:** The app fetches detailed metadata about the scanned medication including:
    - **Side Effects:** Lists common and rare side effects.
    - **Drug Interactions:** Identifies potential interactions with other medications.
    - **Dosage Directions:** Provides clear dosage instructions.
    - **Adverse Reactions & Warnings:** Alerts users to possible adverse effects and important warnings.
  - **Real-Time Updates:** Ensures that the information is current and in line with FDA standards.

- **LLama3-8b Integration:**  
  - **Personalized Analysis:** Leverages Llama3-8b to interpret and translate raw medication data into user-friendly information.  
  - **Contextual Alerts:** Merges FDA API results with the user’s profile to generate tailored interaction warnings and dosage recommendations.
  - **Bullet-Point Summaries:** Delivers concise, accessible insights to help users understand medication details quickly.

- **Intuitive UI & Data Handling:**  
  - **Structured Data Format:** Processes and organizes the retrieved information into a structured JSON format, simplifying further integration.  
  - **Interaction Alerts:** Automatically triggers alerts if any dangerous interactions are detected with the user’s existing medications.  
  - **User-Friendly Interface:** Presents all data in an intuitive, easy-to-navigate UI that enhances user experience and safety.

---

## Tech Stack

- **Swift:** iOS app development  
- **Python:** FastAPI backend & ML algorithms  
- **Apple VisionKit:** OCR for medication image processing  
- **Llama3-8b:** Advanced text analysis and personalized data translation  
- **AWS SageMaker:** Cloud hosting and scalable application management  
- **FDA API:** Access to up-to-date, reliable drug information  
- **MongoDB:** Secure management of user profiles and medication databases
