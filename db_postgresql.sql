CREATE TABLE alunos (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    data_nascimento DATE
);

CREATE TABLE professores (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    especialidade VARCHAR(100)
);

CREATE TABLE materias (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    id_professor INT REFERENCES professores(id)
);

CREATE TABLE notas (
    id SERIAL PRIMARY KEY,
    id_aluno INT REFERENCES alunos(id),
    id_materia INT REFERENCES materias(id),
    nota NUMERIC(4,2),
    data_avaliacao DATE
);

INSERT INTO professores (nome, especialidade) VALUES
('Carlos Silva', 'Matemática'),
('Mariana Costa', 'Português'),
('Pedro Santos', 'História'),
('Fernanda Lima', 'Geografia'),
('Lucas Oliveira', 'Física');

INSERT INTO materias (nome, id_professor) VALUES
('Matemática', 1),
('Português', 2),
('História', 3),
('Geografia', 4),
('Física', 5);

INSERT INTO alunos (nome, data_nascimento) VALUES
('Ana Souza', '2008-05-14'),
('Bruno Alves', '2007-11-23'),
('Camila Ferreira', '2009-03-05'),
('Diego Martins', '2008-08-17'),
('Eduarda Rocha', '2007-12-01'),
('Felipe Gomes', '2008-09-30'),
('Gabriela Dias', '2009-01-20'),
('Henrique Pinto', '2007-07-07'),
('Isabela Nunes', '2008-04-25'),
('João Pedro Lima', '2009-02-10');

INSERT INTO notas (id_aluno, id_materia, nota, data_avaliacao) VALUES
(1, 1, 8.5, '2024-06-01'),
(1, 2, 7.0, '2024-06-02'),
(1, 3, 9.0, '2024-06-03'),
(2, 1, 6.5, '2024-06-01'),
(2, 4, 7.8, '2024-06-04'),
(3, 2, 9.2, '2024-06-02'),
(3, 3, 8.7, '2024-06-03'),
(4, 1, 7.0, '2024-06-01'),
(4, 5, 6.9, '2024-06-05'),
(5, 4, 8.0, '2024-06-04'),
(5, 5, 7.5, '2024-06-05'),
(6, 1, 8.8, '2024-06-01'),
(6, 2, 9.1, '2024-06-02'),
(7, 3, 6.4, '2024-06-03'),
(7, 4, 7.7, '2024-06-04'),
(8, 5, 9.0, '2024-06-05'),
(9, 1, 7.5, '2024-06-01'),
(9, 2, 8.3, '2024-06-02'),
(9, 3, 7.9, '2024-06-03'),
(10, 4, 8.4, '2024-06-04'),
(10, 5, 9.2, '2024-06-05');
