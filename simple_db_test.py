from dotenv import load_dotenv
load_dotenv()
from db.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) as count FROM llms'))
    count = result.fetchone()[0]
    print(f'Database connected. LLMs table has {count} records.')

    # Insert a test record
    conn.execute(text("INSERT INTO llms (id, name, description, base_model) VALUES ('test-123', 'Test LLM', 'Test Description', 'test-base')"))
    conn.commit()

    # Query it back
    result = conn.execute(text("SELECT name FROM llms WHERE id = 'test-123'"))
    name = result.fetchone()[0]
    print(f'Retrieved test record: {name}')

    # Clean up
    conn.execute(text("DELETE FROM llms WHERE id = 'test-123'"))
    conn.commit()

    print('Database connectivity and persistence verified!')
