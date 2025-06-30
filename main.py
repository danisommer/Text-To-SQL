import mysql.connector
import psycopg2
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import logging as transformers_logging
from dotenv import load_dotenv
from tabulate import tabulate
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

# Carrega variáveis de ambiente e configura o registro
load_dotenv()
transformers_logging.set_verbosity_error()

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.db_type = None
        self.schema = {}
        self.column_types = {}
        self.foreign_keys = {}
        self.primary_keys = {}
        self.table_comments = {}
        
    def connect(self, db_type: str, host: str, user: str, password: str, database: str, port: int = None) -> bool:
        """Conecta a um banco de dados MySQL ou PostgreSQL"""
        self.db_type = db_type.lower()
        try:
            if self.db_type == 'mysql':
                self.connection = mysql.connector.connect(
                    host=host, user=user, password=password, database=database, port=port if port else 3306
                )
            elif self.db_type == 'postgresql':
                self.connection = psycopg2.connect(
                    host=host, user=user, password=password, dbname=database, port=port if port else 5432
                )
            else:
                print(f"Tipo de banco de dados não suportado: {db_type}")
                return False
            
            print(f"Conectado ao banco {database}")
            self.load_schema()
            return True
        except Exception as e:
            print(f"Erro ao conectar: {e}")
            return False
    
    def disconnect(self) -> None:
        """Fecha a conexao com o banco de dados"""
        if self.connection:
            self.connection.close()
            print("Conexão fechada")
    
    def load_schema(self) -> None:
        """Carrega o esquema do banco de dados"""
        if not self.connection:
            print("Não conectado a nenhum banco de dados")
            return
        
        cursor = self.connection.cursor()
        
        try:
            if self.db_type == 'mysql':
                # Carrega tabelas e colunas
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                
                for table in tables:
                    # Obtém colunas
                    cursor.execute(f"DESCRIBE {table}")
                    table_info = cursor.fetchall()
                    columns = [col[0] for col in table_info]
                    column_types = {col[0]: col[1] for col in table_info}
                    self.schema[table] = columns
                    self.column_types[table] = column_types
                    
                    # Obtém chaves primárias
                    cursor.execute(f"SHOW KEYS FROM {table} WHERE Key_name = 'PRIMARY'")
                    pk_info = cursor.fetchall()
                    self.primary_keys[table] = [pk[4] for pk in pk_info]
                    
                    # Obtém chaves estrangeiras
                    cursor.execute(f"""
                        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                        WHERE TABLE_NAME = '{table}' AND REFERENCED_TABLE_NAME IS NOT NULL
                    """)
                    fk_info = cursor.fetchall()
                    self.foreign_keys[table] = [(fk[0], fk[1], fk[2]) for fk in fk_info]
                    
                    # Obtém comentários da tabela
                    cursor.execute(f"""
                        SELECT TABLE_COMMENT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}'
                    """)
                    comment_info = cursor.fetchone()
                    self.table_comments[table] = comment_info[0] if comment_info and comment_info[0] else ""
                    
            elif self.db_type == 'postgresql':
                # Carrega tabelas
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                tables = [table[0] for table in cursor.fetchall()]
                
                for table in tables:
                    # Obtém colunas
                    cursor.execute(f"""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = '{table}'
                        ORDER BY ordinal_position
                    """)
                    table_info = cursor.fetchall()
                    columns = [col[0] for col in table_info]
                    column_types = {col[0]: f"{col[1]} {'NULL' if col[2]=='YES' else 'NOT NULL'}" for col in table_info}
                    self.schema[table] = columns
                    self.column_types[table] = column_types
                    
                    # Obtém chaves primárias
                    cursor.execute(f"""
                        SELECT column_name FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                        WHERE tc.table_name = '{table}' AND tc.constraint_type = 'PRIMARY KEY'
                    """)
                    pk_info = cursor.fetchall()
                    self.primary_keys[table] = [pk[0] for pk in pk_info]
                    
                    # Obtém chaves estrangeiras
                    cursor.execute(f"""
                        SELECT kcu.column_name, ccu.table_name, ccu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                        JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
                        WHERE tc.table_name = '{table}' AND tc.constraint_type = 'FOREIGN KEY'
                    """)
                    fk_info = cursor.fetchall()
                    self.foreign_keys[table] = [(fk[0], fk[1], fk[2]) for fk in fk_info]
                    
                    # Obtém comentários da tabela
                    cursor.execute(f"""
                        SELECT obj_description(oid) FROM pg_class WHERE relname = '{table}' AND relkind = 'r'
                    """)
                    comment_info = cursor.fetchone()
                    self.table_comments[table] = comment_info[0] if comment_info and comment_info[0] else ""
                    
        except Exception as e:
            print(f"Erro ao carregar esquema: {e}")
        finally:
            cursor.close()
    
    def execute_query(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """Executa consulta SQL e retorna resultados com nomes das colunas"""
        if not self.connection:
            print("Não conectado ao banco")
            return [], []
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            return results, column_names
        except Exception as e:
            print(f"Erro ao executar consulta: {e}")
            return [], []
        finally:
            cursor.close()


class TextToSQL:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.schema = db_connection.schema
        self.column_types = db_connection.column_types
        self.foreign_keys = db_connection.foreign_keys
        self.primary_keys = db_connection.primary_keys
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.model_loaded = False
        
        # Enhanced patterns with better Portuguese support for rule-based parsing
        self.patterns = {
            'select_all': r'(?:mostre|exibir?|listar?|quais|qual|todos|todas|ver|visualizar).*?(?:de|da|do|das|dos|na|no|nas|nos)\s+([\w\s]+)',
            'count': r'(?:quantos?|quantas?|contar?|conte|número|numero|total).*?(?:de|da|do|das|dos|em|na|no|nas|nos)\s+([\w\s]+)',
            'average': r'(?:média|media|médias|medias).*?(?:de|da|do|das|dos)\s+([\w\s]+).*?(?:em|na|no|nas|nos|do|de)\s+([\w\s]+)',
            'max': r'(?:maior|máximo|maximo|máxima|maxima).*?(?:de|da|do|das|dos)\s+([\w\s]+)',
            'min': r'(?:menor|mínimo|minimo|máxima|minima).*?(?:de|da|do|das|dos)\s+([\w\s]+)',
            'where_condition': r'(?:onde|que tenha|contendo|com)\s+([\w\s]+)\s+(?:igual a|é|eh|=|como|seja)\s+([\w\s\'\"]+)',
            'greater_than': r'(?:maior(?:es)? que|acima de|superior(?:es)? a)\s+(\d+(?:\.\d+)?)',
            'less_than': r'(?:menor(?:es)? que|abaixo de|inferior(?:es)? a)\s+(\d+(?:\.\d+)?)',
            'greater_equal': r'(?:maior(?:es)? ou igual a|a partir de)\s+(\d+(?:\.\d+)?)',
            'less_equal': r'(?:menor(?:es)? ou igual a|até)\s+(\d+(?:\.\d+)?)',
            'join_indicators': r'(?:junto com|com|relacionado|vinculado|associado)',
            'group_by': r'(?:agrupar|agrupado|por cada|para cada)\s+([\w\s]+)',
            'order_by': r'(?:ordenar|ordenado|classificar)\s+(?:por\s+)?([\w\s]+)(?:\s+(crescente|decrescente|asc|desc))?'
        }
        
        # Try to load the AI model
        try:
            self.load_model()
        except Exception as e:
            print(f"Aviso: Não foi possível carregar o modelo de IA: {e}")
            print("Usando método baseado em regras como alternativa.")
            self.model_loaded = False
    
    def load_model(self):
        """Carrega o modelo Llama 3.2"""
        try:
            print("Carregando modelo Llama 3.2...")
            
            model_name = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
            use_cpu_only = os.getenv("USE_CPU_ONLY", "false").lower() == "true"
            device_map = "cpu" if use_cpu_only or not torch.cuda.is_available() else "auto"
            torch_dtype = torch.float32 if use_cpu_only or not torch.cuda.is_available() else torch.float16
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Cria pipeline para geração de texto
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            print("Modelo carregado com sucesso")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            raise Exception(f"Falha ao carregar modelo: {e}")

    def format_schema_for_prompt(self):
        """Formata o esquema do banco de dados para o prompt"""
        schema_text = "DATABASE SCHEMA:\n\n"
        
        # Adiciona visão geral das tabelas
        schema_text += "TABLES:\n"
        for table_name in self.schema.keys():
            schema_text += f"- {table_name}\n"
        schema_text += "\n"
        
        # Adiciona detalhes para cada tabela
        for table_name, columns in self.schema.items():
            schema_text += f"TABLE: {table_name}\n"
            
            # Chaves primárias
            primary_keys = self.db.primary_keys.get(table_name, [])
            if primary_keys:
                schema_text += f"Primary Key(s): {', '.join(primary_keys)}\n"
            
            # Colunas com tipos
            schema_text += "Columns:\n"
            for column in columns:
                col_type = self.column_types.get(table_name, {}).get(column, "unknown")
                schema_text += f"  - {column}: {col_type}\n"
            
            # Chaves estrangeiras
            foreign_keys = self.db.foreign_keys.get(table_name, [])
            if foreign_keys:
                schema_text += "Foreign Keys:\n"
                for fk_col, ref_table, ref_col in foreign_keys:
                    schema_text += f"  - {fk_col} -> {ref_table}.{ref_col}\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def generate_sql_with_llama(self, text: str) -> str:
        """Gera SQL usando o modelo Llama"""
        if not self.model_loaded or self.pipe is None:
            raise Exception("Modelo Llama não carregado")
        
        schema_text = self.format_schema_for_prompt()
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert SQL assistant. Convert Portuguese natural language queries to SQL using the provided database schema.

{schema_text}

GUIDELINES:
- Respond with ONLY the SQL query, no explanations
- Use proper JOIN syntax when accessing multiple tables
- Use the exact table and column names from the schema
- Handle Portuguese text and accents correctly

<|eot_id|><|start_header_id|>user<|end_header_id|>

Convert this Portuguese query to SQL: {text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        response = self.pipe(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        if response and len(response) > 0:
            generated_text = response[0]["generated_text"]
            return self.extract_sql_from_response(generated_text)
        else:
            raise Exception("Resposta vazia do modelo")

    def extract_sql_from_response(self, response: str) -> str:
        """Extrai a consulta SQL da resposta do modelo"""
        # Remove qualquer formatação markdown e extrai o SQL
        sql_match = re.search(r"SQL query:\s*(.*?)(?:$|```)", response, re.DOTALL)
        
        if sql_match:
            sql = sql_match.group(1).strip()
            sql = re.sub(r"```sql|```", "", sql).strip()
            return sql
        
        # Alternativa: encontra qualquer parte que pareça SQL
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'SELECT' in line.upper() or 'WITH' in line.upper():
                sql_lines = []
                for j in range(i, len(lines)):
                    if lines[j].strip() and not lines[j].startswith('```'):
                        sql_lines.append(lines[j].strip())
                    if j > i and (lines[j].startswith('```') or lines[j].startswith('Note:') or lines[j].startswith('Explanation:')):
                        break
                return ' '.join(sql_lines)
        
        # Alternativa: retorna tudo depois de "SQL query:"
        if "SQL query:" in response:
            return response.split("SQL query:")[1].strip()
        
        return response.strip()
    
    def similarity(self, a: str, b: str) -> float:
        """Calcula similaridade entre duas strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_table_by_keyword(self, text: str) -> str:
        """Encontra a tabela mais provavel com melhor matching"""
        text = text.lower()
        best_match = None
        best_score = 0
        
        # Mapeamento de palavras comuns para tabelas
        table_aliases = {
            'aluno': 'alunos',
            'estudante': 'alunos',
            'estudantes': 'alunos',
            'professor': 'professores',
            'docente': 'professores',
            'docentes': 'professores',
            'materia': 'materias',
            'disciplina': 'materias',
            'disciplinas': 'materias',
            'nota': 'notas',
            'avaliacao': 'notas',
            'avaliacoes': 'notas',
            'avaliação': 'notas',
            'avaliações': 'notas'
        }
        
        # Primeiro, verifica aliases
        for alias, table in table_aliases.items():
            if alias in text and table in self.schema:
                return table
        
        # Depois verifica similaridade com nomes de tabelas
        for table in self.schema.keys():
            similarity_score = self.similarity(table, text)
            if similarity_score > 0.6 and similarity_score > best_score:
                best_match = table
                best_score = similarity_score
            
            # Verifica se a tabela está presente no texto
            if table.lower() in text:
                score = len(table) * 2
                if score > best_score:
                    best_match = table
                    best_score = score
                        
        return best_match

    def find_column_by_keyword(self, table: str, text: str, context: str = '') -> Optional[str]:
        """Encontra a coluna mais provavel com contexto melhorado"""
        if not table or table not in self.schema:
            return None
            
        text = text.lower()
        columns = self.schema[table]
        best_match = None
        best_score = 0
        
        # Mapeamento de palavras para colunas comuns
        column_aliases = {
            'nome': 'nome',
            'nomes': 'nome',
            'data': 'data_nascimento' if 'nascimento' in text else 'data_avaliacao',
            'nascimento': 'data_nascimento',
            'idade': 'data_nascimento',
            'especialidade': 'especialidade',
            'area': 'especialidade',
            'nota': 'nota',
            'notas': 'nota',
            'pontuacao': 'nota',
            'pontuações': 'nota',
            'avaliacao': 'data_avaliacao',
            'avaliação': 'data_avaliacao'
        }
        
        # Verifica aliases primeiro
        for alias, col_name in column_aliases.items():
            if alias in text and col_name in columns:
                return col_name
        
        # Verifica similaridade
        for col in columns:
            if col.lower() in text:
                score = len(col) * 2
                if score > best_score:
                    best_match = col
                    best_score = score
            
            similarity_score = self.similarity(col, text)
            if similarity_score > 0.7 and similarity_score > best_score:
                best_match = col
                best_score = similarity_score
        
        return best_match
    
    def extract_year(self, text: str) -> Optional[str]:
        """Extrai o ano do texto da consulta com mais padrões"""
        year_patterns = [
            r'(?:de |em |no |na |ano de |ano )(\d{4})',
            r'(\d{4})',
            r'(?:desde |a partir de )(\d{4})',
            r'(?:até |ate )(\d{4})'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2100:  # Validação básica de ano
                    return str(year)
        return None
    
    def detect_joins(self, text: str) -> List[Dict]:
        """Detecta necessidade de JOINs baseado no contexto"""
        joins = []
        text = text.lower()
        
        # Mapeamento de relacionamentos conhecidos
        relationships = {}
        
        # Constrói mapeamento de relacionamentos a partir das chaves estrangeiras
        for table, fks in self.db.foreign_keys.items():
            for fk_col, ref_table, ref_col in fks:
                key = (table, ref_table)
                relationships[key] = (fk_col, ref_col)
        
        # Detecta se múltiplas tabelas são mencionadas
        mentioned_tables = []
        for table in self.schema.keys():
            if table.lower() in text or any(alias in text for alias in ['aluno', 'professor', 'materia', 'nota']):
                if table == 'alunos' and ('aluno' in text or 'estudante' in text):
                    mentioned_tables.append(table)
                elif table == 'professores' and ('professor' in text or 'docente' in text):
                    mentioned_tables.append(table)
                elif table == 'materias' and ('materia' in text or 'disciplina' in text):
                    mentioned_tables.append(table)
                elif table == 'notas' and ('nota' in text or 'avaliacao' in text):
                    mentioned_tables.append(table)
        
        # Gera JOINs necessários
        for i, table1 in enumerate(mentioned_tables):
            for table2 in mentioned_tables[i+1:]:
                if (table1, table2) in relationships:
                    col1, col2 = relationships[(table1, table2)]
                    joins.append({
                        'table1': table1,
                        'table2': table2,
                        'column1': col1,
                        'column2': col2
                    })
                elif (table2, table1) in relationships:
                    col2, col1 = relationships[(table2, table1)]
                    joins.append({
                        'table1': table2,
                        'table2': table1,
                        'column1': col2,
                        'column2': col1
                    })
        
        return joins
        
    def parse_multiple_conditions(self, text: str, table: str) -> List[Dict]:
        """Analisa múltiplas condições WHERE conectadas por 'e'/'and'"""
        conditions = []
        
        # Divide o texto por conectores
        parts = re.split(r'\s+e\s+|\s+and\s+', text)
        
        for part in parts:
            part = part.strip()
            
            # Condição de igualdade
            where_match = re.search(r'([\w\s]+)\s+(?:é|eh|=|igual a|seja)\s+([\w\s\'\"]+)', part)
            if where_match:
                field_name = where_match.group(1).strip()
                value = where_match.group(2).strip().strip('\'"')
                column = self.find_column_by_keyword(table, field_name)
                if column:
                    conditions.append({'type': 'equality', 'column': column, 'value': value})
                continue
            
            # Filtro por ano específico
            year_match = re.search(r'ano\s+(?:de\s+)?(?:nascimento\s+)?(?:é|eh|=)\s+(\d{4})', part)
            if year_match:
                year = year_match.group(1)
                date_columns = [col for col in self.schema[table] 
                               if any(date_term in col.lower() for date_term in 
                                     ['data', 'date', 'dt', 'ano', 'year'])]
                if date_columns:
                    conditions.append({'type': 'year', 'column': date_columns[0], 'value': year})
                continue
            
            # Operadores de comparação
            for op_name, pattern in [
                ('greater_than', r'(?:maior(?:es)? que|acima de|superior(?:es)? a)\s+(\d+(?:\.\d+)?)'),
                ('less_than', r'(?:menor(?:es)? que|abaixo de|inferior(?:es)? a)\s+(\d+(?:\.\d+)?)'),
                ('greater_equal', r'(?:maior(?:es)? ou igual a|a partir de)\s+(\d+(?:\.\d+)?)'),
                ('less_equal', r'(?:menor(?:es)? ou igual a|até)\s+(\d+(?:\.\d+)?)')
            ]:
                match = re.search(pattern, part)
                if match:
                    value = match.group(1).strip()
                    column = self.find_column_by_keyword(table, part) or 'nota'
                    operator_map = {
                        'greater_than': '>',
                        'less_than': '<',
                        'greater_equal': '>=',
                        'less_equal': '<='
                    }
                    conditions.append({
                        'type': 'comparison',
                        'column': column, 
                        'operator': operator_map[op_name], 
                        'value': value
                    })
                    break
        
        return conditions

    def parse_query_with_rules(self, text: str) -> Dict:
        """Analisa consulta usando método baseado em regras (sem IA)"""
        original_text = text
        text = text.lower()
        query_type = 'select'
        table = self.find_table_by_keyword(text)
        
        if not table:
            return {'error': 'Não foi possível identificar a tabela na consulta'}
            
        query_info = {'type': query_type, 'table': table}
        
        # Detecta tipo de operação com melhor busca por colunas
        if re.search(r'média|media|médias|medias', text):
            query_info['type'] = 'average'
            # Busca coluna após "média de" ou similar
            avg_match = re.search(r'média\s+(?:de\s+|da\s+|das\s+|do\s+|dos\s+)?([\w\s]+)', text)
            if avg_match:
                column_hint = avg_match.group(1).strip()
                query_info['column'] = self.find_column_by_keyword(table, column_hint, 'average')
            if not query_info.get('column'):
                query_info['column'] = 'nota'  # Default
                
        elif re.search(r'quant[oa]s|contar|conte|número|numero|total', text):
            query_info['type'] = 'count'
            
        elif re.search(r'máximo|maximo|maior', text) and not re.search(r'maior(?:es)? que|acima de|superior(?:es)? a', text):
            query_info['type'] = 'max'
            # Busca coluna após "maior" ou "máximo"
            max_match = re.search(r'(?:maior|máximo|maximo)\s+(?:de\s+|da\s+|das\s+|do\s+|dos\s+)?([\w\s]+)', text)
            if max_match:
                column_hint = max_match.group(1).strip()
                query_info['column'] = self.find_column_by_keyword(table, column_hint, 'max')
            if not query_info.get('column'):
                # Se não encontrou coluna específica, tenta detectar contexto
                if 'nota' in text or 'avaliacao' in text or 'pontuacao' in text:
                    query_info['column'] = 'nota'
                else:
                    query_info['column'] = 'nota'  # Default
                    
        elif re.search(r'mínimo|minimo|menor', text) and not re.search(r'menor(?:es)? que|abaixo de|inferior(?:es)? a', text):
            query_info['type'] = 'min'
            # Busca coluna após "menor" ou "mínimo"
            min_match = re.search(r'(?:menor|mínimo|minimo)\s+(?:de\s+|da\s+|das\s+|do\s+|dos\s+)?([\w\s]+)', text)
            if min_match:
                column_hint = min_match.group(1).strip()
                query_info['column'] = self.find_column_by_keyword(table, column_hint, 'min')
            if not query_info.get('column'):
                query_info['column'] = 'nota'  # Default
        
        # Detecta JOINs necessários
        joins = self.detect_joins(text)
        if joins:
            query_info['joins'] = joins
            
        # Detecta condições WHERE com suporte a múltiplas condições
        where_part = ""
        if ' onde ' in text:
            where_part = text.split(' onde ', 1)[1]
        elif ' que ' in text and ('tenha' in text or 'tem' in text):
            where_part = text.split(' que ', 1)[1]
        elif ' com ' in text:
            where_part = text.split(' com ', 1)[1]
            
        if where_part:
            conditions = self.parse_multiple_conditions(where_part, table)
            if conditions:
                query_info['where'] = {'conditions': conditions}
        
        # Se não encontrou condições estruturadas, tenta padrões simples
        if 'where' not in query_info:
            # Condição simples de igualdade
            where_match = re.search(self.patterns['where_condition'], text)
            if where_match:
                field_name = where_match.group(1).strip()
                value = where_match.group(2).strip().strip('\'"')
                column = self.find_column_by_keyword(table, field_name)
                
                if column:
                    query_info['where'] = {'conditions': [{'type': 'equality', 'column': column, 'value': value}]}
            
            # Operadores de comparação únicos
            for op_name, pattern in [
                ('greater_than', self.patterns['greater_than']),
                ('less_than', self.patterns['less_than']),
                ('greater_equal', self.patterns['greater_equal']),
                ('less_equal', self.patterns['less_equal'])
            ]:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).strip()
                    column = self.find_column_by_keyword(table, text) or 'nota'
                    operator_map = {
                        'greater_than': '>',
                        'less_than': '<',
                        'greater_equal': '>=',
                        'less_equal': '<='
                    }
                    query_info.setdefault('where', {})
                    query_info['where']['conditions'] = [{
                        'type': 'comparison',
                        'column': column, 
                        'operator': operator_map[op_name], 
                        'value': value
                    }]
                    break
                    
            # Filtro por ano genérico
            year = self.extract_year(text)
            if year and 'where' not in query_info:
                date_columns = [col for col in self.schema[table] 
                               if any(date_term in col.lower() for date_term in 
                                     ['data', 'date', 'dt', 'ano', 'year'])]
                
                if date_columns:
                    query_info['where'] = {'conditions': [{'type': 'year', 'column': date_columns[0], 'value': year}]}
        
        # Detecta ORDER BY
        order_match = re.search(self.patterns['order_by'], text)
        if order_match:
            order_column = self.find_column_by_keyword(table, order_match.group(1))
            if order_column:
                direction = 'ASC'
                if order_match.group(2):
                    direction = 'DESC' if order_match.group(2).lower() in ['decrescente', 'desc'] else 'ASC'
                query_info['order_by'] = {'column': order_column, 'direction': direction}
                
        return query_info
    
    def generate_sql_from_rules(self, query_info: Dict) -> str:
        """Gera SQL a partir das informações da consulta baseada em regras"""
        if 'error' in query_info:
            return query_info['error']
        
        table = query_info['table']
        query_type = query_info['type']
        
        # Constrói parte SELECT
        if query_type == 'select':
            select_clause = "SELECT *"
        elif query_type == 'count':
            select_clause = "SELECT COUNT(*)"
        elif query_type == 'average':
            column = query_info.get('column', 'nota')
            select_clause = f"SELECT AVG({column})"
        elif query_type == 'max':
            column = query_info.get('column', 'nota')
            select_clause = f"SELECT MAX({column})"
        elif query_type == 'min':
            column = query_info.get('column', 'nota')
            select_clause = f"SELECT MIN({column})"
        
        # Constrói parte FROM com JOINs
        from_clause = f"FROM {table}"
        if 'joins' in query_info:
            for join in query_info['joins']:
                from_clause += f" JOIN {join['table2']} ON {join['table1']}.{join['column1']} = {join['table2']}.{join['column2']}"
        
        sql = f"{select_clause} {from_clause}"
        
        # Adiciona condições WHERE com nova estrutura
        if 'where' in query_info and 'conditions' in query_info['where']:
            conditions = []
            
            for condition in query_info['where']['conditions']:
                if condition['type'] == 'equality':
                    column = condition['column']
                    value = condition['value']
                    if value.isdigit():
                        conditions.append(f"{column} = {value}")
                    else:
                        conditions.append(f"{column} LIKE '%{value}%'")
                        
                elif condition['type'] == 'comparison':
                    column = condition['column']
                    operator = condition['operator']
                    value = condition['value']
                    conditions.append(f"{column} {operator} {value}")
                        
                elif condition['type'] == 'year':
                    year_col = condition['column']
                    year_val = condition['value']
                    
                    if self.db.db_type == 'mysql':
                        conditions.append(f"YEAR({year_col}) = {year_val}")
                    else:
                        conditions.append(f"EXTRACT(YEAR FROM {year_col}) = {year_val}")
                        
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
                
        # Suporte à estrutura WHERE antiga (compatibilidade)
        elif 'where' in query_info:
            conditions = []
            
            if 'column' in query_info['where']:
                column = query_info['where']['column']
                value = query_info['where']['value']
                if value.isdigit():
                    conditions.append(f"{column} = {value}")
                else:
                    conditions.append(f"{column} LIKE '%{value}%'")
            
            if 'comparison' in query_info['where']:
                comp_info = query_info['where']['comparison']
                column = comp_info['column']
                operator = comp_info['operator']
                value = comp_info['value']
                conditions.append(f"{column} {operator} {value}")
                    
            if 'year' in query_info['where']:
                year_col = query_info['where']['year']['column']
                year_val = query_info['where']['year']['value']
                
                if self.db.db_type == 'mysql':
                    conditions.append(f"YEAR({year_col}) = {year_val}")
                else:
                    conditions.append(f"EXTRACT(YEAR FROM {year_col}) = {year_val}")
                    
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
        
        # Adiciona ORDER BY
        if 'order_by' in query_info:
            order_info = query_info['order_by']
            sql += f" ORDER BY {order_info['column']} {order_info['direction']}"
                
        return sql
    
    def parse_query(self, text: str) -> Dict:
        """Usa Llama para analisar a consulta em linguagem natural, com fallback para regras"""
        if self.model_loaded:
            try:
                # Tenta usar a IA primeiro
                sql = self.generate_sql_with_llama(text)
                return {
                    "raw_sql": sql,
                    "parsed": True
                }
            except Exception as e:
                print(f"Erro na geração do SQL com IA: {e}")
                print("Usando método baseado em regras como fallback.")
                # Se falhar, tenta o método baseado em regras
                return self.parse_query_with_rules(text)
        else:
            # Se o modelo não foi carregado, usa diretamente o método baseado em regras
            return self.parse_query_with_rules(text)
    
    def generate_sql(self, query_info: Dict) -> str:
        """Gera SQL a partir das informações da consulta"""
        if "error" in query_info:
            return query_info["error"]
        
        if "raw_sql" in query_info:
            return query_info["raw_sql"]
        
        # Se não tiver SQL pronto da IA, gera SQL a partir das regras
        return self.generate_sql_from_rules(query_info)

def display_results(results: List[Tuple], column_names: List[str]) -> None:
    """Exibe resultados com formatação melhorada"""
    if not results:
        print("Nenhum resultado encontrado")
        return
    
    print(f"\nEncontrados {len(results)} resultado(s)")
    print("=" * 50)
    
    if len(results) <= 20:
        print(tabulate(results, headers=column_names, tablefmt="grid"))
    else:
        print(f"Mostrando primeiros 20 de {len(results)} resultados:")
        print(tabulate(results[:20], headers=column_names, tablefmt="grid"))
        print(f"\n... e mais {len(results) - 20} resultado(s)")

def print_usage_guide():
    """Guia de uso do sistema"""
    guide = """
SISTEMA TEXT-TO-SQL COM LLAMA 3.2

EXEMPLOS DE CONSULTAS:
   "mostre todos os alunos"
   "quantos professores existem"
   "média de notas dos alunos"
   "maior nota registrada"
   "alunos com notas maiores que 8"
   "professores de matemática"

COMANDOS:
   'guia' - Mostra este guia
   'sair' - Encerra o programa
"""
    print(guide)

def show_tables_summary(schema: Dict[str, List[str]], column_types: Dict[str, Dict[str, str]], 
                       primary_keys: Dict[str, List[str]] = None, 
                       foreign_keys: Dict[str, List[Tuple[str, str, str]]] = None) -> None:
    """Exibe resumo melhorado das tabelas com relacionamentos"""
    if not schema:
        print("\nNenhuma tabela disponível")
        return
        
    print("\nESTRUTURA DO BANCO:")
    
    for table in schema.keys():
        columns = schema[table]
        types = column_types.get(table, {})
        pks = primary_keys.get(table, []) if primary_keys else []
        fks = foreign_keys.get(table, []) if foreign_keys else []
        
        print(f"\n{table.upper()} ({len(columns)} colunas)")
        print("-" * 40)
        
        # Mostra chaves primárias
        if pks:
            print(f"   🔑 Chave(s) primária(s): {', '.join(pks)}")
        
        # Mostra colunas
        for col in columns[:6]:  # Mostra até 6 colunas
            col_type = types.get(col, 'unknown')
            pk_marker = " [PK]" if col in pks else ""
            fk_marker = ""
            for fk_col, ref_table, ref_col in fks:
                if fk_col == col:
                    fk_marker = f" -> {ref_table}.{ref_col}"
                    break
            print(f"   • {col} ({col_type}){pk_marker}{fk_marker}")
        
        if len(columns) > 6:
            print(f"   ... e mais {len(columns) - 6} coluna(s)")
        
        # Mostra relacionamentos de chave estrangeira
        if fks:
            print("   🔗 Relacionamentos:")
            for fk_col, ref_table, ref_col in fks[:3]:  # Mostra até 3 FKs
                print(f"     {fk_col} -> {ref_table}.{ref_col}")

def main():
    print("SISTEMA TEXT-TO-SQL COM LLAMA 3.2")
    
    db = DatabaseConnection()
    
    print("\nSelecione o tipo de banco:")
    print("1. MySQL")
    print("2. PostgreSQL")
    
    choice = input("\nSua escolha [1]: ").strip() or "1"
    
    db_type = "mysql" if choice == "1" else "postgresql" if choice == "2" else "mysql"
    
    if choice not in ["1", "2"]:
        print("Opção inválida. Usando MySQL.")
        db_type = "mysql"
    
    host = "localhost"
    database = input(f"\nNome do banco: ").strip()
    
    if db_type == "mysql":
        port = int(os.getenv("MYSQL_PORT", 3306))
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
    else:
        port = int(os.getenv("POSTGRESQL_PORT", 5432))
        user = os.getenv("POSTGRESQL_USER", "postgres")
        password = os.getenv("POSTGRESQL_PASSWORD", "postgres")
        
    print(f"\nConectando ao {db_type.upper()} em {host}:{port}...")
    
    if not db.connect(db_type, host, user, password, database, port):
        print(f"\nFalha na conexão com '{database}'")
        return
    
    print("\nIniciando sistema Text-to-SQL...")
    text_to_sql = TextToSQL(db)
    
    # Choose processing method
    use_ai = False
    if text_to_sql.model_loaded:
        print("\nSelecione o método de processamento de consultas:")
        print("1. Baseado em regras (mais rápido, menos flexível)")
        print("2. Modelo de IA Llama 3.2 (mais preciso, mais lento)")
        method_choice = input("\nSua escolha [2]: ").strip() or "2"
        use_ai = method_choice == "2"
        
        if use_ai:
            print("\nUsando modelo Llama 3.2 para processamento de consultas!")
        else:
            print("\nUsando método baseado em regras para processamento de consultas.")
    else:
        print("\nModelo Llama 3.2 não disponível, usando método baseado em regras.")
    
    print("\nSistema pronto! Digite suas consultas em português.")
    
    while True:
        show_tables_summary(db.schema, db.column_types, db.primary_keys, db.foreign_keys)
        
        print(f"\nDigite sua consulta (ou 'sair' para sair, 'guia' para ajuda):")
        query = input("> ").strip()
        
        if query.lower() in ('sair', 's', 'exit', 'quit', 'q'):
            print("Até logo!")
            break
        elif query.lower() in ('guia', 'guide', 'help', 'ajuda'):
            print_usage_guide()
            continue
        elif not query:
            continue
            
        print(f"\nProcessando: '{query}'")
        
        try:
            # Use the selected method based on user choice
            if use_ai and text_to_sql.model_loaded:
                # Use AI-based method
                sql = text_to_sql.generate_sql_with_llama(query)
                print(f"\nSQL gerado (IA): {sql}")
            else:
                # Use rules-based method
                query_info = text_to_sql.parse_query_with_rules(query)
                sql = text_to_sql.generate_sql_from_rules(query_info)
                print(f"\nSQL gerado (regras): {sql}")
            
            results, column_names = db.execute_query(sql)
            display_results(results, column_names)
        except Exception as e:
            print(f"Erro: {e}")
    
    db.disconnect()

if __name__ == "__main__":
    main()