-- =============================================================================
-- VAP MODELS TABLE
-- Registro de modelos de segmentación personalizados por sponsor almacenados en S3.
--
-- Estructura de ruta S3:
--   model_route = "{bucket}/{pais}/{sponsor}/{model}"
--   Ejemplo    : "aihubmodelos/Colombia/Bancolombia/model_output_col_multitag"
--
-- El campo time_priors_route apunta al JSON de priors temporales del modelo.
-- Por convención se guarda dentro del mismo directorio de modelo, pero puede
-- ser una ruta independiente.
-- =============================================================================

CREATE TABLE IF NOT EXISTS vap_models (
    id                INT            NOT NULL AUTO_INCREMENT
        COMMENT 'PK autoincremental',

    sponsor_id        INT            NOT NULL
        COMMENT 'FK lógica a marketing_campaigns.sponsor_id',

    model_route       VARCHAR(1024)  NOT NULL
        COMMENT 'Ruta completa en S3: {bucket}/{pais}/{sponsor}/{model_dir}',

    time_priors_route VARCHAR(1024)      NULL
        COMMENT 'Ruta S3 al archivo time_priors.json. NULL = se asume dentro de model_route/',

    model_name        VARCHAR(255)   NOT NULL
        COMMENT 'Nombre descriptivo/versión del modelo (ej. model_output_col_multitag_v2)',

    upload_date       DATETIME       NOT NULL  DEFAULT CURRENT_TIMESTAMP
        COMMENT 'Fecha y hora en que el modelo fue subido a S3',

    tested            TINYINT(1)     NOT NULL  DEFAULT 0
        COMMENT '1 = modelo validado y aprobado para producción, 0 = experimental',

    PRIMARY KEY (id),

    -- Índice principal de consulta: para un sponsor, el modelo más reciente primero,
    -- con preferencia por modelos probados.
    INDEX idx_sponsor_tested_date (sponsor_id, tested DESC, upload_date DESC),

    -- Índice para búsquedas por ruta (útil para verificar duplicados).
    INDEX idx_model_route (model_route(255))

) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Modelos de segmentación personalizados por sponsor en S3';


-- =============================================================================
-- QUERY DE REFERENCIA: obtener el mejor modelo para un sponsor dado.
--
-- Prioridad:
--   1. Modelos marcados como tested=1 (validados).
--   2. Entre ellos (o entre todos si no hay ninguno tested), el más reciente.
--
-- Uso:
--   Reemplaza :sponsor_id por el valor concreto.
-- =============================================================================
--
-- SELECT
--     id,
--     sponsor_id,
--     model_route,
--     time_priors_route,
--     model_name,
--     upload_date,
--     tested
-- FROM vap_models
-- WHERE sponsor_id = :sponsor_id
-- ORDER BY tested DESC, upload_date DESC
-- LIMIT 1;


-- =============================================================================
-- EJEMPLOS DE INSERCIÓN
-- =============================================================================
--
-- INSERT INTO vap_models (sponsor_id, model_route, time_priors_route, model_name, upload_date, tested)
-- VALUES (
--     42,
--     'aihubmodelos/Colombia/Bancolombia/model_output_col_multitag_v3',
--     NULL,          -- time_priors.json vive dentro del directorio del modelo
--     'model_output_col_multitag_v3',
--     '2025-03-10 14:30:00',
--     1              -- ya validado
-- );
--
-- INSERT INTO vap_models (sponsor_id, model_route, time_priors_route, model_name, upload_date, tested)
-- VALUES (
--     17,
--     'aihubmodelos/Peru/TelefonicaPeru/model_output_telefonica_v1',
--     'aihubmodelos/Peru/TelefonicaPeru/model_output_telefonica_v1/time_priors.json',
--     'model_output_telefonica_v1',
--     NOW(),
--     0              -- aún en pruebas
-- );
