import React from 'react';

const PhraseInputs = ({ phrase1, phrase2, onPhrase1Change, onPhrase2Change }) => {
    return (
        <div className="input-group">
            <input
                type="text"
                className="input-box"
                id="phrase1"
                value={phrase1}
                onChange={onPhrase1Change}
                placeholder="Enter start phrase"
            />
            <input
                type="text"
                className="input-box"
                id="phrase2"
                value={phrase2}
                onChange={onPhrase2Change}
                placeholder="Enter end phrase"
            />
        </div>
    );
};

export default PhraseInputs; 