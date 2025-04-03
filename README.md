# Corpus-Graph Document Translation
Corpus informed, graph-based document-level translation

## Overview
This project investigates potential improvements to machine translation by combining three approaches:
- Corpus-informed translation using domain-specific monolingual corpora
- Graph-based translation leveraging terminological knowledge graphs
- Document-level translation based upon specifications that moves beyond traditional sentence segmentation

## Project Goals
- Explore whether domain-specific corpora in translation/localization can improve translation quality
- Test if terminological knowledge graphs can enhance consistency and accuracy
- Test if automatic translation systems can be trained to customize translation output to project specifications
- Move beyond sentence-level segmentation to a whole document (whole context) based approach to enable more natural expression in target languages
- Integrate standards-based evaluation methodologies to assess translation improvements

## Project Structure

```
corpus-graph-document-translation/
├── backend/
│   ├── .env # Environment variables file (not tracked by git)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── database.py           # MongoDB connection setup
│   │   ├── models.py
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── documents.py      # API endpoints
│   └── main.py                   # FastAPI app entry point
├── frontend/                     # GitHub Pages implementation
│   ├── css/
│   ├── js/
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
Traditional machine translation and Computer-Assisted Translation (CAT) tools typically process text segment by segment. This project explores whether considering broader document context, combined with domain-specific knowledge, can produce more natural and accurate translations.

## Installation & Usage

### Google Cloud Project (GCP)
**Name**: Corpora Graph Auto Translation
**Buckets**
- auto-translator-corpora

**auto-translator-corpora bucket: directory structure**

```
auto-translator-corpora/
├── gai/ # Generative AI corpora
│   ├── en-intl/ # English content in variants besides U.S. English
│   │   ├── metadata/
│   │   ├── processed/
│   │   └── submissions/
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

[To be developed as additional components are developed]

## Contributing
Contributions and feedback are welcome. Please feel free to submit issues and pull requests.

## License
This work is licensed under a GNU Affero General Public License (AGPL) v3.0.

### Citation
If you use this project in your research, please cite it as:

APA:
Brandt, A. (2025). Corpus-graph-document-translation: Corpus informed, graph-based document-level translation. GitHub. https://github.com/alainamb/corpus-graph-document-translation

BibTeX:
@misc{corpusgraphtranslation2025,
    author = {Brandt, Alaina},
    title = {Corpus-graph-document-translation: Corpus informed, graph-based document-level translation},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/alainamb/corpus-graph-document-translation}}
}

## Contact
Contact me on Github: @alainamb