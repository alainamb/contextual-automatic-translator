# Contextual Automatic Translator
The Contextual Automatic Translator produces bespoke translations through the integration of specifications, specialized corpora and terminological knowledge graphs. This translator also take a full-document (rather than a sentence-by-sentence) approach to translation.

## Overview
This project investigates potential improvements to machine translation by combining these approaches:
- Corpus-informed translation using domain-specific monolingual corpora
- Graph-based translation leveraging terminological knowledge graphs
- Specifications-based automatic translation with customized outputs depending on audience, purpose, etc.
- Document-level translation that moves beyond the traditional sentence segmentation approach

## Project Goals
- Explore whether domain-specific corpora can improve translation quality
- Test if terminological knowledge graphs can enhance consistency and accuracy
- Test if automatic translation systems can be trained to customize translation output to project specifications
- Move beyond sentence-level segmentation to a whole document (whole context) based approach to enable more natural expression in target languages
- Integrate standards-based evaluation methodologies to assess translation improvements

## Project Structure

```
contextual-automatic-translator/
├── backend/
│   ├── .env # Environment variables file (not tracked by git)
│   ├── app/
│   │   ├── experiments/
│   │       └── clustering # Python for testing clustering algorithms to be integrated into similarity scoring among documents
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── documents.py      # API endpoints
│   │   ├── __init__.py
│   │   ├── database.py           # MongoDB connection setup
│   │   ├── models.py
│   └── main.py                   # FastAPI app entry point
├── frontend/                     # GitHub Pages implementation
│   ├── css/
│   ├── js/
│   ├── submissions.html
│   ├── validation.html
│   ├── kb-search.html
│   ├── references.html
│   ├── js/
│   │   └── navbar.js    
├── index.html                    
├── .gitignore                    # Excludes backend/.env
├── LICENSE
└── README.md                     # Explain GCP usage here
```

## Current Status
This project is in early developmental stages. The implementation will progress as follows:
1. Corpora integration
- MongoDB Atlas
- Python FastAPI backend
- Backend deployed to cloud platform (such as Railway)
2. Ongoing frontend development (HTML/CSS/JavaScript/React)
3. Knowledge graph integration
4. Backend integration with LangChain
5. Evaluation methodology implementation

## Research Context
Traditional machine translation and Computer-Assisted Translation (CAT) tools typically process text segment by segment. This project explores whether considering broader document context, combined with domain-specific knowledge, can produce more natural and accurate translations.

## Standards-Based Approach

**Language Code Standardization** 
The ISO 639-2 standard (two-letter codes) provides codes for 400-500 languages, while ISO 639-3 (three-letter codes) supports over 7,600 distinct language identifiers. This project proactively adopts ISO 639-3 three-letter language codes.

This expansion is crucial because:
   - Supports more granular language variant representation
   - Prepares for comprehensive global language coverage
     * Enables precise linguistic variant specification
     * Supports emerging and minority language documentation

**Quality Evaluation Standards**
This project incorporates translation quality evaluation methodologies outlined in two forthcoming ASTM F43 standards:
- The MQM core typology for identifying individual errors within automatic translation outputs
- The HQuest axes of correspondence and readability for rating the quality of the translation as a whole document

## Technical Stack
- Frontend: HTML5, CSS3, JavaScript, React
- Backend: Python, LangChain, FastAPI
- Data Management: Google Cloud Project, MongoDB Atlas
- Deployment: Railway

### Google Cloud Project (GCP)
**Name**: Corpora Graph Auto Translation

**Buckets**
- auto-translator-corpora

**auto-translator-corpora bucket: directory structure**

```
auto-translator-corpora/
├── gai/ # Generative AI corpora
│   ├── en-intl/ # English content in variants besides U.S. English
│   │   ├── submissions/       # Initial upload location (status: "new")
│   │   ├── validated/         # Validated documents
│   │   ├── rejected/          # Rejected documents
│   │   ├── deprecated/        # Deprecated documents (previously valid)
│   │   ├── processed/         # Processed text files (of validated docs)
│   │   └── metadata/          # JSON metadata files
│   ├── en-us/ # Prioritized English content
│   │   └── [Similar structure]
│   ├── es-intl/ # Spanish content in variants besides LATAM Spanish
│   │   └── [Similar structure]
│   ├── en-latam/ # Prioritized; Subfolders to be created as needed
│   │   └── [Similar structure]
├── tl/ # Translation & Localization corpora
│   └── [Similar structure]
```

### MongoDB Atlas
**Cluster name**: Auto-Translator

**Databases**
- Auto-Translator
  - Collections
    - document_metadata

## Installation & Usage
[To be developed as additional components are developed]

## Contributing
Contributions and feedback are welcome. Please feel free to submit issues and pull requests.

## License
This work is licensed under a GNU Affero General Public License (AGPL) v3.0.

### Citation
If you use this project in your research, please cite it as:

APA:
Brandt, A. (2025). Contextual Automatic Translator: Integrating Specifications, Specialized Corpora & Terminological Knowledge into a Whole Document Approach. GitHub. https://github.com/alainamb/corpus-graph-document-translation

BibTeX:
@misc{contextualautomatictranslator2025,
    author = {Brandt, Alaina},
    title = {Contextual Automatic Translator: Integrating Specifications, Specialized Corpora & Terminological Knowledge into a Whole Document Approach},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/alainamb/contextual-automatic-translator}}
}

## Contact
Contact me on Github: @alainamb