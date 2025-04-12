// JS code by Dr. Sanchez via MIT professional education course on web programming
class References extends HTMLElement{
    constructor(){
        super();

        const referenceCategory = this.getAttribute('referenceCategory');
        const referenceTitle = this.getAttribute('referenceTitle');
        const authors = this.getAttribute('authors');
        const pubYear = this.getAttribute('pubYear');
        const fullWorkTitle = this.getAttribute('fullWorkTitle');
        const publisher = this.getAttribute('publisher');
        const referenceURL = this.getAttribute('referenceURL');
        const description = this.getAttribute('description');

        this.innerHTML = `
        <div class="reference-card">
            <h3 class="reference-card-header">${referenceTitle}</h3>
            <div class="reference-card-body">
                <p class="reference-card-text">
                <table class="table">
                    <tr>
                        <td class="tbd">Title:</td>
                        <td>${referenceTitle}</td>
                    </tr>
                    <tr>
                        <td class="tbd">Authors:</td>
                        <td>${authors}</td>
                    </tr>
                    <tr>
                        <td class="tbd">Year:</td>
                        <td>${pubYear}</td>
                    </tr>                         
                    <tr>
                        <td class="tbd">Published in:</td>
                        <td>${fullWorkTitle}</td>
                    </tr>
                    <tr>
                        <td class="tbd">Publisher:</td>
                        <td>${publisher}</td>
                    </tr>
                    <tr>
                        <td class="tbd">URL:</td>
                        <td>${referenceURL}</td>
                    </tr>                              
                </table>
                </p>
            </div>
        </div>
        `;
    }
}

customElements.define('baker-FourUniversals', References);
customElements.define('haywood-genre', References);
customElements.define('mqm-core', References);