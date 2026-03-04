Docstring for process_transcripts_consent_evaluator
    :param transcripts_path: is the path to the transcripts
    :type transcripts_path: str
    :param script: Is the script text, it depends on the use case
    :type script: str
    :param mode: This function mode, can be 'all' or 'sample', defaults to 'all'. 'sample' processes only a subset of files and its useful for testing.
    :type mode: str
    :param only_n: If mode is 'sample', this parameter indicates how many files to process, defaults to 2.
    :type only_n: int
    :return: Dataframe with the evaluation of the consent evaluation for each transcript.
    :rtype: DataFrame


an example of its use would be:

from consent_evaluator.consent_evaluator import process_transcripts_consent_evaluator

df_resultado = process_transcripts_consent_evaluator(transcripts_path = path_de_prueba, script = GUION_OPERACION)
