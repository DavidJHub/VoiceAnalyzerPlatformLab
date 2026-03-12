

def test_presidio():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    print("antes provider")

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": "es_core_news_md"}],
    })

    print("antes create_engine")
    nlp_engine = provider.create_engine()
    print("engine ok")
    print(nlp_engine)
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["es"])
    print("analyzer ok")
    print(analyzer)

    anonymizer = AnonymizerEngine()
    print("anonymizer ok")
    print(anonymizer)

test_presidio()

def test_w_text():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": "es_core_news_md"}],
    })
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["es"])
    anonymizer = AnonymizerEngine()

    text = "Mi nombre es Juan y mi correo es juan@gmail.com y mi celular es 3001234567"

    results = analyzer.analyze(
        text=text,
        language="es",
        entities=["PERSON", "URL", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE"],
    )

    print(results)

    out = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )
    print(out.text)

test_w_text()
