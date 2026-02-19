import { createRouter, createWebHistory } from 'vue-router'

const routes = [
    {
        path: '/',
        name: 'root',
        component: () => import('@/views/Root.vue'),
        children: [

        ]
    }
]

const router = createRouter({
    history: createWebHistory("/"),
    routes,
})

export default router
